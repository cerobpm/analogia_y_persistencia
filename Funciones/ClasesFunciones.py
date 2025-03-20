# -*- coding: utf-8 -*-
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta, datetime, date
from scipy.optimize import fsolve

import requests
import json
import pytz
import dateutil.parser

config_file = open("config.json")
config = json.load(config_file)
config_file.close()


# a6
def readSerie(series_id,timestart=None,timeend=None,tipo="puntual",use_proxy=False):
    params = {}
    if timestart is not None and timeend is not None:
        params = {
            "timestart": timestart if isinstance(timestart,str) else timestart.isoformat(),
            "timeend": timeend if isinstance(timestart,str) else timeend.isoformat()
        }
    testVar="%s/obs/%s/series/%i" % (config["api"]["url"],tipo, series_id)
    print(testVar)
    response = requests.get("%s/obs/%s/series/%i" % (config["api"]["url"], tipo, series_id),
        params = params,
        headers = {'Authorization': 'Bearer ' + config["api"]["token"]},
        proxies = config["proxy_dict"] if use_proxy else None
    )
    if response.status_code != 200:
        raise Exception("request failed: %s" % response.text)
    json_response = response.json()
    return json_response

def observacionesListToDataFrame(data: list):
    if len(data) == 0:
        raise Exception("empty list")
    data = pd.DataFrame.from_dict(data)
    data.index = data["timestart"].apply(tryParseAndLocalizeDate)
    data.sort_index(inplace=True)
    return data[["valor",]]

def tryParseAndLocalizeDate(date_string,timezone='UTC'): # America/Argentina/Buenos_Aires'):
    date = dateutil.parser.isoparse(date_string) if isinstance(date_string,str) else date_string
    if date.tzinfo is None or date.tzinfo.utcoffset(date) is None:
        try:
            date = pytz.timezone(timezone).localize(date)
        except pytz.exceptions.NonExistentTimeError:
            print("NonexistentTimeError: %s" % str(date))
            return None
    else:
        date = date.astimezone(pytz.timezone(timezone))
    return date

def roundDownDate(date,timeInterval,timeOffset=None):
    if timeInterval.microseconds == 0:
        date = date.replace(microsecond=0)
    if timeInterval.seconds % 60 == 0:
        date = date.replace(second=0)
    if timeInterval.seconds % 3600 == 0:
        date = date.replace(minute=0)
    if timeInterval.seconds == 0 and timeInterval.days >= 1:
        date = date.replace(hour=0)
        if timeOffset is not None:
            date = date + timeOffset
    return date

def createDatetimeSequence(datetime_index : pd.DatetimeIndex=None, 
                           timeInterval=timedelta(days=1), 
                           timestart=None, timeend=None, timeOffset=None):
    #Fechas desde timestart a timeend con un paso de timeInterval
    #data: dataframe con index tipo datetime64[ns, America/Argentina/Buenos_Aires]
    #timeOffset sólo para timeInterval n days
    if datetime_index is None and (timestart is None or timeend is None):
        raise Exception("Missing datetime_index or timestart+timeend")
    timestart = timestart if timestart is not None else datetime_index.min()
    timestart = roundDownDate(timestart,timeInterval,timeOffset)
    timeend = timeend if timeend  is not None else datetime_index.max()
    timeend = roundDownDate(timeend,timeInterval,timeOffset)
    return pd.date_range(start=timestart, end=timeend, freq=pd.DateOffset(days=timeInterval.days, hours=timeInterval.seconds // 3600, minutes = (timeInterval.seconds // 60) % 60))

def serieRegular(data : pd.DataFrame, 
                 timeInterval : timedelta, 
                 timestart=None, timeend=None, timeOffset=None, column="valor", 
                 interpolate=True, interpolation_limit=1):
    # genera serie regular y rellena nulos interpolando
    df_regular = pd.DataFrame(index = createDatetimeSequence(data.index, timeInterval, timestart, timeend, timeOffset))
    df_regular.index.rename('timestart', inplace=True)	 
    df_join = df_regular.join(data, how = 'outer')
    if interpolate:
        # Interpola
        df_join[column] = df_join[column].interpolate(method='time',limit=interpolation_limit,limit_direction='both')
    df_regular = df_regular.join(df_join, how = 'left')
    return df_regular

def prono2serie(df,main_colname="h_sim",members={'e_pred_01':'p01','e_pred_99':'p99'},series_id=3398):
    df_simulado = df.copy().reset_index()
    # df_simulado['fecha'] = df_simulado['fecha'].dt.tz_localize("America/Argentina/Buenos_Aires") # timezone.localize(df_simulado['fecha'])
    column_mapper = {
        'month': 'timestart'}
    column_mapper[main_colname] = 'valor'
    df_para_upsert = df_simulado[['month',main_colname]].rename(axis=1, mapper=column_mapper,inplace = False)
    # print(df_para_upsert)
    df_para_upsert['qualifier'] = 'main'
    for member in members:
        column_mapper = { 'month': 'timestart'}
        column_mapper[member] = "valor"
        df_para_upsert = df_para_upsert.append(df_simulado[['month',member]].rename(axis=1, mapper=column_mapper), ignore_index=True)
        df_para_upsert['qualifier'].fillna(value=members[member],inplace=True)
    df_para_upsert['timeend'] = df_para_upsert['timestart']  # .map(lambda a : a.isoformat())
    return {
                'series_table': 'series',
                'series_id': series_id,
                'pronosticos': json.loads(df_para_upsert.to_json(orient='records',date_format='iso'))
        }

def GuardaEnBBDD(series,fecha_emision,cal_id_0):
    '''{
        "forecast_date": "string",
        "series": [
            {
            "series_table": "series",
            "series_id": 0,
            "pronosticos": [
                {
                "timestart": "2023-04-10T17:38:08.365Z",
                "timeend": "2023-04-10T17:38:08.365Z",
                "valor": 0,
                "qualifier": "string"
                }
            ]
            }
        ]
        } '''
    
    with open("config.json") as f:
        config = json.load(f)

    apiLoginParams = config["api"]

    def uploadProno(data,cal_id,responseOutputFile):
        response = requests.post(
            apiLoginParams["url"] + '/sim/calibrados/' + str(cal_id) + '/corridas',
            data=json.dumps(data),
            headers={'Authorization': 'Bearer ' + apiLoginParams["token"], 'Content-type': 'application/json'},
        )
        print("prono upload, response code: " + str(response.status_code))
        print("prono upload, reason: " + response.reason)
        if(response.status_code == 200):
            if(responseOutputFile):
                outresponse = open(responseOutputFile,"w")
                outresponse.write(json.dumps(response.json()))
                outresponse.close()

    def prono2json(series,forecast_date=datetime.now()):
        return {
            'forecast_date': forecast_date.isoformat(),
            'series': series
        }

    def outputjson(data,outputfile):
        output = open(outputfile,'w')
        output.write(json.dumps(data))
        output.close()

    def uploadPronoSeries(series,cal_id=cal_id_0,forecast_date=datetime.now(),outputfile=None,responseOutputFile=None):
        data = prono2json(series,forecast_date)
        if(outputfile):
            outputjson(data,outputfile)
            uploadProno(data,cal_id,responseOutputFile)

    uploadPronoSeries(series,
                      cal_id=cal_id_0,
                      forecast_date=fecha_emision,
                      outputfile='salida.json',
                      responseOutputFile='salidaResponse.json')


###############################################################################################################################################################

class Obj_Serie:
    # Crea un objeto serie
    # if not isinstance(extraObs,type(None)):
    def __init__(self,params,timestart=None,timeend=None):
        self.id = params["id_serie"]
        self.name = params["name"]
        self.var = params["variable"]
        self.timestart = timestart
        self.timeend = timeend
        self.origen = params["origen"]
    
    def SeriesReadBBDD(self):# Carga datos
        serie_i = readSerie(self.id,self.timestart,self.timeend)
        self.serie = observacionesListToDataFrame(serie_i["observaciones"])

    def toCSV(self,serie_i,ruta,sufijo='0'):
        return serie_i.to_csv(ruta+self.name+'_'+str(self.id)+'_'+sufijo+'.csv')
    
    def SeriesReadCSV(self,ruta,sufijo='0'):
        self.serie = pd.read_csv(ruta+self.name+'_'+str(self.id)+'_'+sufijo+'.csv')#,index_col=0)    #, sep=';', encoding = "ISO-8859-1"
        self.serie.index = self.serie["timestart"].apply(tryParseAndLocalizeDate)
        del self.serie["timestart"]

    def SerieRegulariza(self,param):
        self.serie_reg = serieRegular(self.serie,timeInterval=timedelta(hours=param['timeHInterval']),interpolation_limit=param['limIterp'])
        self.NNaN = self.serie_reg['valor'].isna().sum()
        return self.serie_reg
    
    def CreaVariablesTemporales(self,df):   # Variables Temporales
        df.insert(0, 'year', df.index.year)
        df.insert(1, 'month', df.index.month)
        df.insert(2, 'day', df.index.day)
        df.insert(3, 'yrDay', df.index.dayofyear)
        df.insert(4, 'wkDay', df.index.isocalendar().week)

    def ResampleSerie(self,df,vent_temp,var):
        #print(df.head())
        df_resamp = df.groupby(["year",vent_temp]).agg( Variable=(var, 'mean'), 
                                                        Count=(var, 'count')).reset_index()  # { var: "mean",var:"count"}
        return df_resamp

    def PlotFaltantes(self,df,variable,nanvalue=-5): # Para ver faltantes
        df[variable] = df[variable].replace(np.nan,nanvalue)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(df.index, df[variable],'-',linewidth=2)
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.title(self.name)
        #plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()
        plt.close()

        df[variable] = df[variable].replace(nanvalue,np.nan)

    def descript_estadisticos(self,df,variable):# Estadistica Descriptiva
        '''
        Las estadísticas descriptivas incluyen aquellas que resumen 
        la tendencia central, la dispersión y la forma de la distribución 
        de un conjunto de datos, excluyendo los valores de NaN.
        '''
        print('Variable: ',variable)
        media = df[variable].mean()
        mediana = df[variable].median()
        moda = df[variable].mode()
        print("""
            Media: %d
            Mediana: %d
            Moda: %d
        """ % (media,mediana,moda))
        print(df.describe())
    
    # Histograma de Frecuencias
    def HistoVariable(self,df,variable,round_val = 100):
        print('Análisis de frecuencia')
        # round_val se usa para redondear el label de las barras
        from matplotlib.ticker import PercentFormatter

        num_of_bins = round(5*np.log10(len(df))+1)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)

        n, bins, patches = ax.hist(df[variable], edgecolor='black', weights=np.ones_like(df[variable])*100 / len(df[variable]), bins=num_of_bins, rwidth=0.9,color='#607c8e')
        ax.yaxis.set_major_formatter(PercentFormatter())

        bins = [round(item/round_val)*round_val for item in bins]
        plt.xticks(bins)
        plt.tick_params(axis='y', labelsize=14)
        plt.tick_params(axis='x', labelsize=12,rotation=45)
        plt.ylabel('Frecuencia de aparición', size=18)

        
        plt.xlabel('Caudal [m'+r'$^3$'+'/s]', size=18)
        plt.grid(axis='y', alpha=0.75, linewidth=0.3)
        plt.show()

    def Permanencia(self,df,var):
        df_i = df.sort_values(by='Caudal',ascending=False).reset_index(drop=True)
        df_i = df_i[['Caudal',]].dropna()
        df_i['rank'] = df_i.index + 1
        df_i['p_sup'] = df_i['rank'] * 100 / len(df_i)
        print(df_i)

        
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)
        ax.plot(df_i.p_sup, df_i[var],'-',linewidth=2)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Porcentaje de superación [%]', size=18)
        plt.ylabel('Caudal [m'+r'$^3$'+'/s]', size=18)
        plt.grid(alpha=0.75, linewidth=0.5)
        plt.show()

class Obj_Serie_v2:
    # Crea un objeto serie
    # if not isinstance(extraObs,type(None)):
    def __init__(self,id_serie,params):
        self.id = id_serie
        self.var = params["variable"]
        if self.var == 'Caudal':
            self.label_text = 'Caudal [m'+r'$^3$'+'/s]'
        if self.var == 'Nivel':
            self.label_text = 'Nivel [m]'
        self.origen = params["origen"]    

        if "name" in params:
            self.name = params["name"]
        else:
            self.name = None
        if "timestart" in params:
            self.timestart = params["timestart"]
        else:
            self.timestart = None
        if "timeend" in params:
            self.timeend = params["timeend"]
        else:
            self.timeend = None
        if 'id_serie_relleno' in params:
            self.id_serie_relleno = params["id_serie_relleno"]
        else: 
            self.id_serie_relleno = None
        
        # Carga la Serie
        if self.origen == 'BBDD':
            self.SeriesReadBBDD()                     # Lee la serie de la BBDD
            #ObjSerie_i.toCSV(ObjSerie_i.serie,file_path)   # Guarda serie en CSV
        if self.origen == 'CSV':
            self.SeriesReadCSV(params['file_path'])             # Lee la serie de un csv
        
        self.serie = self.serie.rename(columns={'valor':self.var})
        if self.serie_relleno is not None:
            self.serie_relleno = self.serie_relleno.rename(columns={'valor':self.var})

    def SeriesReadBBDD(self):# Carga datos
        serie_i = readSerie(self.id,self.timestart,self.timeend)
        self.serie = observacionesListToDataFrame(serie_i["observaciones"])
        if self.id_serie_relleno is not None:
            serie_j = readSerie(self.id_serie_relleno,self.timestart,self.timeend)
            self.serie_relleno = observacionesListToDataFrame(serie_j["observaciones"])
        else:
            self.serie_relleno = None

    def toCSV(self,serie_i,ruta,sufijo='0'):
        return serie_i.to_csv(ruta+self.name+'_'+str(self.id)+'_'+sufijo+'.csv')
    
    def SeriesReadCSV(self,ruta,sufijo='0'):
        self.serie = pd.read_csv(ruta+self.name+'_'+str(self.id)+'_'+sufijo+'.csv')#,index_col=0)    #, sep=';', encoding = "ISO-8859-1"
        self.serie.index = self.serie["timestart"].apply(tryParseAndLocalizeDate)
        del self.serie["timestart"]

    def SerieRegulariza(self,param,fill_nulls=True):
        self.serie_reg = serieRegular(self.serie,timeInterval=timedelta(hours=param['timeHInterval']),interpolation_limit=param['limIterp'],column=self.var)
        self.NNaN = self.serie_reg[self.var].isna().sum()
        if fill_nulls and self.NNaN > 0 and self.serie_relleno is not None:
            serie_relleno_reg = serieRegular(self.serie_relleno,timeInterval=timedelta(hours=param['timeHInterval']),interpolation_limit=param['limIterp'],column=self.var)
            self.serie_reg = serieFillNulls(self.serie_reg, serie_relleno_reg, self.var, self.var)
            
        return self.serie_reg

    def descript_estadisticos(self,df,variable):# Estadistica Descriptiva
        '''
        Las estadísticas descriptivas incluyen aquellas que resumen 
        la tendencia central, la dispersión y la forma de la distribución 
        de un conjunto de datos, excluyendo los valores de NaN.
        '''
        print('Variable: ',variable)
        media = df[variable].mean()
        mediana = df[variable].median()
        moda = df[variable].mode()
        print("""
            Media: %d
            Mediana: %d
            Moda: %d
        """ % (media,mediana,moda))
        print(df.describe())
    
    def removeOutliers(self, df : pd.DataFrame,limite_outliers,column="valor"):
        '''
        remove outliers inline and return outliers data frame
        '''
        print('Detecta Outliers:')
        limit_inf = limite_outliers[0]
        limit_sup = limite_outliers[1]
        print("Limite superior",round(limit_sup,2))
        print("Limite inferior",round(limit_inf,2)) 
        # Finding the Outliers
        outliers = df[( df[column] < limit_inf) | (df[column] > limit_sup)]
        print('Cantidad de outliers: ',len(outliers))
        df[column] = np.where(df[column]>limit_sup,np.nan,
                    np.where(df[column]<limit_inf,np.nan,
                    df[column]))
        return outliers, df

    def detectJumps(self, df : pd.DataFrame,lim_jump,column="valor"):
        '''
        returns jump rows as data frame
        '''
        print('Detecta Saltos:')	
        VecDif = abs(np.diff(df[column].values))
        VecDif = np.append([0,],VecDif)
        coldiff = 'Diff_Valor'
        df[coldiff] = VecDif
        print('Limite Salto (m): ',lim_jump)
        df_saltos = df[df[coldiff] > lim_jump].sort_values(by=coldiff)
        print('Cantidad de Saltos',len(df_saltos))
        del df[coldiff]
        return df_saltos


    def CreaVariablesTemporales(self,df):   # Variables Temporales
        df.insert(0, 'year', df.index.year)
        df.insert(1, 'month', df.index.month)
        df.insert(2, 'day', df.index.day)
        df.insert(3, 'yrDay', df.index.dayofyear)
        df.insert(4, 'wkDay', df.index.isocalendar().week)

    def ResampleSerie(self,df,vent_temp,var):
        #print(df.head())
        df_resamp = df.groupby(["year",vent_temp]).agg( Variable=(var, 'mean'), 
                                                        Count=(var, 'count')).reset_index()  # { var: "mean",var:"count"}
        return df_resamp

    def PlotFaltantes(self,df,variable,nanvalue=-5): # Para ver faltantes
        df[variable] = df[variable].replace(np.nan,nanvalue)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(df.index, df[variable],'-',linewidth=2)
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.title(self.name)
        #plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()
        plt.close()

        df[variable] = df[variable].replace(nanvalue,np.nan)

    def PlotVsCte(self,df,variable,cte=None,desde=None,hasta=None,y_limite=None):
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(df.index, df[variable],'-',linewidth=2)
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)

        if not isinstance(cte,type(None)):    
            for cte_i in cte:
                ax.axhline(y=cte_i, color='r')

        plt.tick_params(axis='both', labelsize=16)
        plt.tick_params(axis='x', labelsize=16,rotation=20)

        plt.xlabel('Fecha', size=18)
        plt.ylabel(self.label_text, size=18)
        
        if not isinstance(desde,type(None)): 
            plt.xlim(desde,hasta)
        if not isinstance(y_limite,type(None)): 
            plt.ylim(y_limite[0],y_limite[1])

        plt.title(self.name)
        #plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()
        plt.close()

    def PlotVarMaxMedMin_Anual(self,df,variable):
        df_anual = df.groupby(["year"]).agg({ variable: ["max","mean","min"]}).reset_index()
        df_anual.set_index(df_anual['year'], inplace=True)
        print(df_anual)
        del df_anual['year']

        df_anual.columns = ['_'.join(col) for col in df_anual.columns.values]
        #df_anual[variable + '_mean'] = df_anual[variable + '_mean'] - df_anual[variable + '_min']
        #df_anual[variable + '_max'] = df_anual[variable + '_max'] - df_anual[variable + '_mean']
        print(df_anual)
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        #df_anual.plot(kind='bar', stacked=True, color=['skyblue', 'cornflowerblue','darkblue' ], ax=ax)

        ax.plot(df_anual.index, df_anual[variable + '_max'],'-',color='darkblue',linewidth=2,label= 'Máximo')
        ax.plot(df_anual.index, df_anual[variable + '_mean'],'-',color='cornflowerblue',linewidth=2,label= 'Medio')        
        ax.plot(df_anual.index, df_anual[variable + '_min'],'-',color='skyblue',linewidth=2,label= 'Mínimo')

        plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
        plt.tick_params(axis='y', labelsize=14)
        plt.tick_params(axis='x', labelsize=9)
        plt.xlabel('Año', size=18)

        plt.ylabel(self.label_text, size=18)
        plt.legend(['Mínimo','Medio','Máximo'],prop={'size':16},loc=0,ncol=1)
        plt.show()
        plt.close()

    def PlotVarMaxMedMin_Mensual(self,df,variable):
        df_mensual = df.groupby(["month"]).agg({ variable: ["max","mean","min"]}).reset_index()
        df_mensual.set_index(df_mensual['month'], inplace=True)
        del df_mensual['month']
        df_mensual.columns = ['_'.join(col) for col in df_mensual.columns.values]
        # df_mensual[variable + '_mean'] = df_mensual[variable + '_mean'] - df_mensual[variable + '_min']
        # df_mensual[variable + '_max'] = df_mensual[variable + '_max'] - df_mensual[variable + '_mean']

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        # df_mensual.plot(kind='bar', stacked=True, color=['skyblue', 'cornflowerblue','darkblue' ], ax=ax)

        ax.plot(df_mensual.index, df_mensual[variable + '_max'],'-',color='darkblue',linewidth=2,label= 'Máximo')
        ax.plot(df_mensual.index, df_mensual[variable + '_mean'],'-',color='cornflowerblue',linewidth=2,label= 'Medio')        
        ax.plot(df_mensual.index, df_mensual[variable + '_min'],'-',color='skyblue',linewidth=2,label= 'Mínimo')

        plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
        plt.tick_params(axis='y', labelsize=14)
        plt.tick_params(axis='x', labelsize=14,rotation=0)
        plt.xlabel('Mes', size=18)
        
        plt.ylabel(self.label_text, size=18)
        plt.legend(['Mínimo','Medio','Máximo'],prop={'size':16},loc=0,ncol=1)
        plt.show()
        plt.close()

    # Histograma de Frecuencias
    def HistoVariable(self,df,variable,round_val = 100):
        print('Análisis de frecuencia')
        # round_val se usa para redondear el label de las barras
        from matplotlib.ticker import PercentFormatter

        num_of_bins = round(5*np.log10(len(df))+1)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)

        n, bins, patches = ax.hist(df[variable], edgecolor='black', weights=np.ones_like(df[variable])*100 / len(df[variable]), bins=num_of_bins, rwidth=0.9,color='#607c8e')
        ax.yaxis.set_major_formatter(PercentFormatter())

        if variable == 'Caudal':
            round_val = 100
        else:
            round_val = 1

        bins = [round(item/round_val)*round_val for item in bins]
        plt.xticks(bins)
        plt.tick_params(axis='y', labelsize=14)
        plt.tick_params(axis='x', labelsize=12,rotation=0)
        plt.ylabel('Frecuencia de aparición', size=18)

        
        plt.xlabel(self.label_text, size=18)
        plt.grid(axis='y', alpha=0.75, linewidth=0.3)
        plt.show()

    def Permanencia(self,df,var,cte=None,y_limite=None):
        df_i = df.sort_values(by=var,ascending=False).reset_index(drop=True)
        df_i = df_i[[var,]].dropna()
        df_i['rank'] = df_i.index + 1
        df_i['p_sup'] = df_i['rank'] * 100 / len(df_i)

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(111)
        ax.plot(df_i.p_sup, df_i[var],'-',linewidth=2)

        if not isinstance(cte,type(None)):
            for cte_i in cte:
                ax.axhline(y=cte_i, color='r')

        if not isinstance(y_limite,type(None)): 
            plt.ylim(y_limite[0],y_limite[1])

        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Porcentaje de superación [%]', size=18)
        plt.ylabel(self.label_text, size=18)
        plt.grid(alpha=0.75, linewidth=0.5)
        plt.show()

    def FiguraSerieBoxPlot(self,dfBoxPlot,v_resamp,var,cte=None,y_limite=None):
        box_plot_data = [dfBoxPlot.loc[dfBoxPlot[v_resamp] == mes_i,var].dropna() for mes_i in np.sort(dfBoxPlot[v_resamp].unique())]
        box_plot_labels = [ 'Enero','Febrero','Marzo','Abril','Mayo','Junio',
                            'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']

        # Arma curvas de Max, Med y Min
        # df_est_mensual = df.groupby([v_resamp]).agg({ var: ["max","mean","min"]}).reset_index()
        #df_est_mensual.set_index(df_est_mensual[v_resamp], inplace=True)
        #del df_est_mensual[v_resamp]
        #df_est_mensual.columns = ['_'.join(col) for col in df_est_mensual.columns.values]

        
        # Grafico
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax.boxplot(box_plot_data,patch_artist=True,labels=box_plot_labels,boxprops={'fill': None})
        
        if not isinstance(cte,type(None)):
            for cte_i in cte:
                ax.axhline(y=cte_i, color='r')

        if not isinstance(y_limite,type(None)): 
            plt.ylim(y_limite[0],y_limite[1])

        plt.title(self.name)    
        plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
        plt.tick_params(axis='y', labelsize=16)
        plt.tick_params(axis='x', labelsize=16,rotation=20)
        plt.xlabel('Mes', size=18)
        plt.ylabel(self.label_text, size=18)
        plt.legend(prop={'size':16},loc=0,ncol=1)
        plt.show()
        plt.close()
    

def IndicadoresDeAjuste(df,VarObs,VarSim,mes_selct,n_var_obs=None,n_var_sim=None):
    Vobs_media = round(np.mean(df[VarObs]),1)
    Vsim_media = round(np.mean(df[VarSim]),1)

    #Nash y Sutcliffe
    F = (np.square(np.subtract(df[VarSim], df[VarObs]))).sum()
    F0 = (np.square(np.subtract(df[VarObs], np.mean(df[VarObs])))).sum()
    E_var = round(100*(F0-F)/F0,3)

    #Coeficiente de correlación (r)
    x = df[VarObs]
    y = df[VarSim]
    r_var = np.round(np.corrcoef(x, y),4)[0, 1]

    #Error cuadratico medio
    df1aux = df.dropna()
    rms_var = np.sqrt(((np.square(np.subtract(df1aux[VarSim], df1aux[VarObs]))).sum())/len(df1aux))
    rms_var = round(rms_var,3)

    #SPEDS
    b = list()
    Qobs_ant = 0
    Qsim_ant = 0
    for index, row in df.iterrows():
        if (row[VarObs] - Qobs_ant)*(row[VarSim] - Qsim_ant) >= 0:
            bi = 1
        else:
            bi = 0
        Qobs_ant = row[VarObs]
        Qsim_ant = row[VarSim]
        b.append(bi)
    SPEDS_var = round(float(100*sum(b)/len(b)),2)

    #Error Volumetrico OJO! esta pasado a volumen diario y no a mensual
    volSim = (df[VarSim].multiply(86400)).sum()
    volObs = (df[VarObs].multiply(86400)).sum()
    ErrorVolumetrico = round(100 * (volSim - volObs) / volObs,1)

    #volSim = round(volSim,1)
    #volObs = round(volObs,1)
    if n_var_obs==None:
        n_var_obs = VarObs
    if n_var_sim==None:
        n_var_sim =VarSim

    df_i = pd.DataFrame({
        'YrObs':[n_var_obs,],
        'MesObs':[mes_selct,],
        'YrSim':[n_var_sim,], 
        'nobs':[len(df)],
        'Vobs_media':[Vobs_media,], 
        'Vsim_media':[Vsim_media,], 
        'Nash':[E_var,], 
        'CoefC':[r_var,],
        'RMSE':[rms_var,], 
        'SPEDS':[SPEDS_var,], 
        'ErrVol':[ErrorVolumetrico,]
        })
    return  df_i

# De BBDD
def cargaObs(serie_id,timestart,timeend):
    response = requests.get(
        'https://alerta.ina.gob.ar/a6/obs/puntual/series/'+str(serie_id)+'/observaciones',
        params={'timestart':timestart,'timeend':timeend},
        headers={'Authorization': 'Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIiwibmFtZSI6IlNhbnRpYWdvIEd1aXp6YXJkaSIsImlhdCI6MTUxNjIzOTAyMn0.YjqQYMCh4AIKSsSEq-QsTGz3Q4WOS5VE-CplGQdInfQ'},)
    json_response = response.json()
    df_obs_i = pd.DataFrame.from_dict(json_response,orient='columns')
    df_obs_i = df_obs_i[['timestart','valor']]
    df_obs_i = df_obs_i.rename(columns={'timestart':'fecha'})

    df_obs_i['fecha'] = pd.to_datetime(df_obs_i['fecha'])
    df_obs_i['valor'] = df_obs_i['valor'].astype(float)

    df_obs_i = df_obs_i.sort_values(by='fecha')
    df_obs_i.set_index(df_obs_i['fecha'], inplace=True)
    
    df_obs_i.index = df_obs_i.index.tz_convert(None)#("America/Argentina/Buenos_Aires")
    df_obs_i.index = df_obs_i.index - timedelta(hours=3)

    df_obs_i['fecha'] = df_obs_i.index
    df_obs_i = df_obs_i.reset_index(drop=True)
    return df_obs_i

def serieFillNulls(data : pd.DataFrame, other_data : pd.DataFrame, column : str="valor", other_column : str="valor", fill_value : float=None, shift_by : int=0, bias : float=0, extend=False, tag_column=None):
    """
    rellena nulos de data con valores de other_data donde coincide el index. Opcionalmente aplica traslado rígido en x (shift_by: n registros) y en y (bias: float)

    si extend=True el índice del dataframe resultante será la unión de los índices de data y other_data (caso contrario será igual al índice de data)
    """
    # logging.debug("before. data.index.name: %s. other_data.index.name: %s" % (data.index.name, other_data.index.name))
    mapper = {}
    mapper[other_column] = "valor_fillnulls"
    how = "outer" if extend else "left"
    if tag_column is not None:
        mapper[tag_column] = "tag_fillnulls"
        data = data.join(other_data[[other_column,tag_column]].rename(mapper,axis=1), how = how)
        data[column] = data[column].fillna(data["valor_fillnulls"].shift(shift_by, axis = 0) + bias)    
        data[tag_column] = data[tag_column].fillna(data["tag_fillnulls"].shift(shift_by, axis = 0))
        if fill_value is not None:
            data[column] = data[column].fillna(fill_value)
            data[tag_column] = data[tag_column].fillna("filled")
        del data["valor_fillnulls"]
        del data["tag_fillnulls"]
    else:
        data = data.join(other_data[[other_column,]].rename(mapper,axis=1), how = how)
        data[column] = data[column].fillna(data["valor_fillnulls"].shift(shift_by, axis = 0) + bias)
        del data["valor_fillnulls"]
        if fill_value is not None:
            data[column] = data[column].fillna(fill_value)
    # logging.debug("after. data.index.name: %s. other_data.index.name: %s" % (data.index.name, other_data.index.name))
    return data

