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

def tryParseAndLocalizeDate(date_string,timezone='America/Argentina/Buenos_Aires'):
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

def createDatetimeSequence(datetime_index : pd.DatetimeIndex=None, timeInterval=timedelta(days=1), timestart=None, timeend=None, timeOffset=None):
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

def serieRegular(data : pd.DataFrame, timeInterval : timedelta, timestart=None, timeend=None, timeOffset=None, column="valor", interpolate=True, interpolation_limit=1):
    # genera serie regular y rellena nulos interpolando
    df_regular = pd.DataFrame(index = createDatetimeSequence(data.index, timeInterval, timestart, timeend, timeOffset))
    df_regular.index.rename('timestart', inplace=True)	 
    df_join = df_regular.join(data, how = 'outer')
    if interpolate:
        # Interpola
        df_join[column] = df_join[column].interpolate(method='time',limit=interpolation_limit,limit_direction='both')
    df_regular = df_regular.join(df_join, how = 'left')
    return df_regular

# Variables Temporales
def CreaVariablesTemporales(df):
    df.insert(0, 'year', df.index.year)
    df.insert(1, 'month', df.index.month)
    df.insert(2, 'day', df.index.day)
    df.insert(3, 'yrDay', df.index.dayofyear)
    df.insert(4, 'wkDay', df.index.isocalendar().week)

def ResampleSerie(df,vent_temp,var):
    df_resamp = df.groupby(["year",vent_temp]).agg( Caudal=(var, 'mean'), 
                                                    Count=(var, 'count')).reset_index()  # { var: "mean",var:"count"}
    return df_resamp

###  Persistencia
def MetodoPersistencia_1Fecha(df_full:'Df con columna de fecha-caudal',
                                var:'Nombre de la variable',
                                mes:'mes seleccionado',year:'year seleccionado',
                                longBusqueda:'long serie hacia atrás',longProno:'longitud del prono',
                                vent_resamp):
    # Busca la fecha seleccionada
    fecha_Obj = df_full.query("year=="+str(year)+" and month=="+str(mes))
    # Toma el caudal de esta fecha
    val_Q = fecha_Obj[var].values[0]
    # Calcula el cuantil de ese caudal para el mes correspondiente.
    ultimo_quantil = (df_full.loc[df_full['month'] == mes,var].dropna()<val_Q).mean()    
    #print(fecha_Obj)
    
    # Toma el id de la fecha seleccionada
    idx_select = fecha_Obj.index.values[0]

    # Arma el Df de datos Obs para la fecha seleccionada
    idx_fecha_fin = idx_select+1
    idx_fecha_inicio = idx_fecha_fin-longBusqueda

    dfObj = df_full[idx_fecha_inicio:idx_fecha_fin].copy()
    dfObj = dfObj[['month',var]]
    dfObj = dfObj.rename(columns={var:year})

    # Arma el Df para el prono. Fecha Selecionada + dias prono
    idx_fecha_fin_prono = idx_fecha_fin+longProno
    index_i = range(idx_fecha_fin, idx_fecha_fin_prono, 1)

    dfProno = pd.DataFrame(index = index_i,columns=['month','VarProno'])
    dfProno['month'] = range(mes+1, mes+1+longProno, 1)
    dfProno.loc[dfProno["month"]  > 12, 'month'] = dfProno.loc[dfProno["month"]  > 12, 'month'] - 12

    # Agrega el Q pronosticado. 
    # Con el cualtil obtenido busca el caudales en los meses siguietne
    for index, row in dfProno.iterrows():
        mes_i = int(row['month'])
        Q_next_month = df_full.loc[df_full['month'] == mes_i,var].quantile(ultimo_quantil)
        dfProno.loc[index,'VarProno']=Q_next_month

    # Arma BoxPlot
    box_plot_data = [df_full.loc[df_full['month'] == mes_i,var].dropna() for mes_i in df_full['month'].unique()]
    box_plot_labels = [ 'Enero','Febrero','Marzo','Abril','Mayo','Junio',
                        'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']

    # Arma curvas de Max, Med y Min
    df_est_mensual = df_full.groupby([vent_resamp]).agg({ var: ["max","mean","min"]}).reset_index()
    df_est_mensual.set_index(df_est_mensual[vent_resamp], inplace=True)
    del df_est_mensual[vent_resamp]
    df_est_mensual.columns = ['_'.join(col) for col in df_est_mensual.columns.values]

    # Grafico
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(dfObj.month, dfObj[year],s=50,c='blue',label='Ultimos '+str(longBusqueda)+' Obs.')
    ax.scatter(dfProno.month, dfProno['VarProno'],s=50,c='red',label='Caudal Pronosticado')
    
    #sns.boxplot(data=df_mensual, x="month", y="Caudal",color="skyblue")
    ax.boxplot(box_plot_data,patch_artist=True,labels=box_plot_labels,boxprops={'fill': None})
            
    plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14,rotation=20)
    plt.xlabel('Mes', size=18)
    plt.ylabel('Caudal [m'+r'$^3$'+'/s]', size=18)
    plt.legend(prop={'size':16},loc=0,ncol=1)
    plt.show()
    plt.close()

def MetodoPersistencia( df_full:'Df con columna de fecha-caudal',
                        var:'Nombre de la variable',
                        mes:'mes seleccionado',year:'year seleccionado',
                        longBusqueda:'long serie hacia atrás',longProno:'longitud del prono'):
    # Busca la fecha seleccionada
    fecha_Obj = df_full.query("year=="+str(year)+" and month=="+str(mes))
    # Toma el caudal de esta fecha
    val_Q = fecha_Obj[var].values[0]
    # Calcula el cuantil de ese caudal para el mes correspondiente.
    ultimo_quantil = (df_full.loc[df_full['month'] == mes,var].dropna()<val_Q).mean()

    # Toma el id de la fecha seleccionada
    idx_select = fecha_Obj.index.values[0]

    # Arma el Df de datos Obs para la fecha seleccionada
    idx_fecha_inico_prono = idx_select+1
    idx_fecha_fin_prono = idx_fecha_inico_prono+longProno

    # Arma el Df para el prono. Fecha Selecionada + dias prono
    dfProno = df_full[idx_fecha_inico_prono:idx_fecha_fin_prono].copy().reset_index()
    dfProno = dfProno[['month','Caudal']]
    dfProno = dfProno.rename(columns={'Caudal':'QObs'})
    dfProno['QProno'] = np.nan
    # Agrega el Q pronosticado. 
    # Con el cualtil obtenido busca el caudales en los meses siguietne
    for index, row in dfProno.iterrows():
        mes_i = int(row['month'])
        Q_next_month = df_full.loc[df_full['month'] == mes_i,var].quantile(ultimo_quantil)
        dfProno.loc[index,'QProno']=Q_next_month
    # Devuelve el Df Con el Prono  
    return dfProno

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





##  Otras

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

