# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import datetime
from datetime import timedelta,datetime
#from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

import timeit
from Funciones.ClasesFunciones import IndicadoresDeAjuste

# Guarda en BBDD

import requests
import json

'''
start = timeit.default_timer()
#Your statements here
stop = timeit.default_timer()
print('Time: ', stop - start) 
'''

map_meses = {1:'Enero',2:'Febrero',3:'Marzo',4:'Abril',5:'Mayo',6:'Junio',7:'Julio',8:'Agosto',9:'Septiembre',10:'Octubre',11:'Noviembre',12:'Diciembre'}

def FiguraSerieBoxPlot(nomEst,df_obs,v_obs,df_sim,v_sim,dfBoxPlot,v_resamp,var,longBusqueda):
    box_plot_data = [dfBoxPlot.loc[dfBoxPlot[v_resamp] == mes_i,var].dropna() for mes_i in np.sort(dfBoxPlot[v_resamp].unique())]
    box_plot_labels = [ 'Enero','Febrero','Marzo','Abril','Mayo','Junio',
                        'Julio','Agosto','Septiembre','Octubre','Noviembre','Diciembre']

    # Arma curvas de Max, Med y Min
    # df_est_mensual = df.groupby([v_resamp]).agg({ var: ["max","mean","min"]}).reset_index()
    #df_est_mensual.set_index(df_est_mensual[v_resamp], inplace=True)
    #del df_est_mensual[v_resamp]
    #df_est_mensual.columns = ['_'.join(col) for col in df_est_mensual.columns.values]

    if var == 'Caudal':
        label_text = 'Caudal [m'+r'$^3$'+'/s]'
    if var == 'Nivel':
        label_text = 'Nivel [m]'
    
    # Grafico
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df_obs[v_resamp], df_obs[v_obs],s=50,c='blue',label='Ultimos '+str(longBusqueda)+' Obs.')
    ax.scatter(df_sim[v_resamp], df_sim[v_sim],s=50,c='red',label='Caudal Pronosticado')
    
    #sns.boxplot(data=df_mensual, x="month", y="Caudal",color="skyblue")
    ax.boxplot(box_plot_data,patch_artist=True,labels=box_plot_labels,boxprops={'fill': None})
    plt.title(nomEst)    
    plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14,rotation=20)
    plt.xlabel('Mes', size=18)
    plt.ylabel(label_text, size=18)
    plt.legend(prop={'size':16},loc=0,ncol=1)
    plt.show()
    plt.close()

def MetodoPersistencia( nomEst:'Nombre de la estacion',
                        df_full:'Df con columna de fecha-variable',
                        var:'Nombre de la variable',
                        mes:'mes seleccionado',
                        year:'year seleccionado',
                        longBusqueda:'long serie hacia atrás',
                        longProno:'longitud del prono',
                        vent_resamp:'Ventana temporal del resampleo',
                        Prono=True,
                        Plot=False):
    
    # Busca la fecha seleccionada
    fecha_Obj = df_full.query("year=="+str(year)+" and "+vent_resamp+"=="+str(mes))
    
    # Toma el id de la fecha seleccionada
    idx_select = fecha_Obj.index.values[0]

    # Toma el caudal de esta fecha
    val_Q = fecha_Obj[var].values[0]

    # Calcula el cuantil de ese caudal para el mes correspondiente.
    df_Base = df_full[:idx_select].copy()   # Filtra datos posteriores a la fecha seleccionada
    ultimo_quantil = (df_Base.loc[df_Base[vent_resamp] == mes,var].dropna()<val_Q).mean()
    #print(fecha_Obj)
    #print('Cuantil cero: ',ultimo_quantil)

    # Index para armar el Df de datos Obs para la fecha seleccionada
    idx_fecha_fin = idx_select+1
    idx_fecha_fin_prono = idx_fecha_fin+longProno
    
    if Prono:
        idx_fecha_inicio = idx_fecha_fin-longBusqueda
        # Arma el Df de datos Obs para la fecha seleccionada
        dfObj = df_full[idx_fecha_inicio:idx_fecha_fin].copy()
        dfObj = dfObj[[vent_resamp,var]]
        dfObj = dfObj.rename(columns={var:year})

        # Arma el Df para el prono. Fecha Selecionada + dias prono
        
        index_i = range(idx_fecha_fin, idx_fecha_fin_prono, 1)
        dfProno = pd.DataFrame(index = index_i,columns=['id',vent_resamp,'VarProno'])
        dfProno[vent_resamp] = range(mes+1, mes+1+longProno, 1)
        
        ### Solo para vent_resamp='month' ####
        dfProno['id']= str(year)+str(mes)
        dfProno['year']= year
        dfProno.loc[dfProno[vent_resamp]  > 12, 'year'] = dfProno.loc[dfProno[vent_resamp]  > 12, 'year'] + 1
        dfProno.loc[dfProno[vent_resamp]  > 12, vent_resamp] = dfProno.loc[dfProno[vent_resamp]  > 12, vent_resamp] - 12

        # Agrega el Q pronosticado.
        # Con el cualtil obtenido busca el caudales en los meses siguietne
        for index, row in dfProno.iterrows():
            mes_i = int(row[vent_resamp])
            Q_next_month = df_Base.loc[df_Base[vent_resamp] == mes_i,var].dropna().quantile(ultimo_quantil)
            dfProno.loc[index,'VarProno']=Q_next_month

            # df_mes_i = df_full.loc[df_full[vent_resamp] == mes_i,var]
            # sns.boxplot(y=df_mes_i)
            # plt.title(mes_i)
            # plt.show()
            # plt.close()

        # Arma BoxPlot
        if Plot:
            FiguraSerieBoxPlot(nomEst,dfObj,year,dfProno,'VarProno',df_full,vent_resamp,var,longBusqueda)
        return dfProno
    
    # Para el Cálculo del error
    else:
        # Arma el Df para el prono. Fecha Selecionada + dias prono
        dfProno = df_full[idx_fecha_fin:idx_fecha_fin_prono].copy().reset_index()
        dfProno = dfProno[[vent_resamp,var]]
        dfProno = dfProno.rename(columns={var:var+'_Obs'})
        dfProno[var+'_Prono'] = np.nan
        
        # Agrega el Q pronosticado. 
        # Con el cualtil obtenido busca el caudales en los meses siguietne
        for index, row in dfProno.iterrows():
            mes_i = int(row[vent_resamp])
            Q_next_month = df_Base.loc[df_Base[vent_resamp] == mes_i,var].dropna().quantile(ultimo_quantil)
            dfProno.loc[index,var+'_Prono']=Q_next_month
        # Devuelve el Df Con el Prono  
        return dfProno

def ErrorXPersistencia(nomEst:      'Nombre Estacion',
                       df:          'Df con columna de fecha-variablo',
                       var:         'Nombre de la variable',
                       l_obs:       'long serie hacia atrás',
                       l_prono:     'longitud del prono',
                       vent_resamp: 'Ventana temporal del resampleo',
                       Plot=True,
                       connBBDD=None):
    print('Calcula Error x Mes: ',nomEst)
    df_errorXMes = pd.DataFrame(columns=['1er Mes','2do Mes','3er Mes'])

    # Corta el df. Saca los dos primeros años y los ultimos 3 meses
    df_clip = df[24:-l_prono]#l_obs

    if connBBDD != None:
        NombreTabla = 'Salidas_Persist'
        cur = connBBDD.cursor()
        cur.execute('DROP TABLE IF EXISTS '+NombreTabla+';')
    
    def calcError(mes,yr):
        mes_selct = int(mes)
        yr_select = int(yr)
        df_prono = MetodoPersistencia(nomEst,df,var,mes_selct,yr_select,l_obs,l_prono,vent_resamp,Prono=False)
        
        if df_prono[var+'_Obs'].isna().sum() == 0:
            df_prono['Dif_Prono'] = df_prono[var+'_Prono'] - df_prono[var+'_Obs']
            df_errorXMes.loc[len(df_errorXMes)] = [df_prono.loc[0,'Dif_Prono'],df_prono.loc[1,'Dif_Prono'],df_prono.loc[2,'Dif_Prono']]

            if connBBDD != None:
                df_prono['nombre'] = nomEst
                df_prono['year'] = yr_select
                df_prono['mes_ant'] = df_prono.index + 1
                df_prono = df_prono[['nombre','year','month','mes_ant',var+'_Obs',var+'_Prono','Dif_Prono']]
                df_prono.to_sql(NombreTabla, con = connBBDD, if_exists='append',index=False)    # Guarda en BBDD
        
    df_clip.apply(
            lambda  row: calcError(row['month'],row['year']),
            axis=1)
    
    if connBBDD != None: connBBDD.commit()

    if Plot:
        box_plot_data = [df_errorXMes[MesAnt] for MesAnt in df_errorXMes.columns]
        box_plot_labels = [MesAnt for MesAnt in df_errorXMes.columns]

        if var == 'Caudal':
            label_text = 'Caudal [m'+r'$^3$'+'/s]'
        if var == 'Nivel':
            label_text = 'Nivel [m]'

        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)

        ax.boxplot(box_plot_data,patch_artist=True,labels=box_plot_labels,boxprops={'fill': 'skyblue'})

        plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
        plt.tick_params(axis='y', labelsize=14)
        plt.tick_params(axis='x', labelsize=14,rotation=0)
        plt.xlabel('Mes', size=18)
        plt.ylabel(label_text, size=18)
        #plt.legend(prop={'size':16},loc=0,ncol=1)
        plt.show()
        plt.close()

### Metodo Analogias.

def TransfDatos(df,var,v_temp,PlotTransf=False):
    #### 1 - Transfroma los datos
    # create log-transformed data
    df['LogVar'] = np.log(df[var])  # Con var < 0 tira warning
    # Normaliza los datos transformados
    df['LogVar_Est'] = np.nan
    for mes in df['month'].unique():
        mes_mean = df.loc[df[v_temp] == mes,'LogVar'].dropna().mean()
        mes_std = df.loc[df[v_temp] == mes,'LogVar'].dropna().std()
        # Normaliza los datos
        df.loc[df['month']==mes,'LogVar_Est'] = (df.loc[df['month']==mes,'LogVar'] - mes_mean)/mes_std
    if PlotTransf:   # plots datos transformados
        fig, axs = plt.subplots(nrows=1, ncols=3)
        #create histograms
        axs[0].hist(df[var], edgecolor='black')
        axs[1].hist(df['LogVar'], edgecolor='black')
        axs[2].hist(df['LogVar_Est'], edgecolor='black')
        #add title to each histogram
        axs[0].set_title('Original Data')
        axs[1].set_title('Log-Transformed Data')
        axs[2].set_title('Log-Transformed-Norm Data')
        plt.show()
        plt.close()

# Calcula indicadores para la fecha seleccionada.
def CalcIndicXFecha(df,year_obj,mes_obj,longBusqueda,longProno):
    columnas = ['YrObs','MesObs','YrSim','nobs', 'Vobs_media', 'Vsim_media', 'Nash', 'CoefC','RMSE', 'SPEDS', 'ErrVol']
    df_indicadores = pd.DataFrame(columns=columnas)
    variable_transf = 'LogVar_Est'
    # Busca la fecha seleccionada
    fecha_Obj = df.query("year=="+str(year_obj)+" and month=="+str(mes_obj))
    # Toma el id de la fecha seleccionada
    idx_select = fecha_Obj.index.values[0]

    # Arma el Df de datos Obs para la fecha seleccionada
    idx_fecha_fin = idx_select+1
    idx_fecha_inicio = idx_fecha_fin-longBusqueda
    dfObj_0 = df[idx_fecha_inicio:idx_fecha_fin].copy()

    if dfObj_0[variable_transf].isna().sum() > 0:
        return False, dfObj_0, 0

    for yr_sim in df['year'].unique():
        if yr_sim == year_obj:continue # <=

        # Arma el Df para comparar con el seleccionado.
        fecha_sim = df.query("year=="+str(yr_sim)+" and month=="+str(mes_obj))
        if len(fecha_sim) == 0:continue
        idx_sim = fecha_sim.index.values[0]
        idx_sim_fin = idx_sim+1
        idx_sim_inicio = idx_sim_fin-longBusqueda

        dfSim = df[idx_sim_inicio:idx_sim_fin].copy()
        dfSim = dfSim[['month',variable_transf]]
        dfSim = dfSim.rename(columns={variable_transf:yr_sim})
        df_union = dfObj_0.merge(dfSim, on='month')

        # Si hay faltantes no calcula los indicadores
        if df_union[yr_sim].isna().sum() > 0: continue
        if len(df_union) == 0: continue

        df_indic_i = IndicadoresDeAjuste(df_union,variable_transf,yr_sim,mes_obj,n_var_obs=year_obj)
        df_indicadores = pd.concat([df_indicadores,df_indic_i])

    # Agrega indicadores
    variables = ['Nash','CoefC','RMSE','SPEDS','ErrVol']
    for var in variables:
        df_indicadores[var+'_norm'] = (df_indicadores[var] - df_indicadores[var].min())/(df_indicadores[var].max()-df_indicadores[var].min())

    df_indicadores['Score'] = df_indicadores['Nash_norm'] + df_indicadores['CoefC_norm'] - df_indicadores['RMSE_norm'] + df_indicadores['SPEDS_norm'] - df_indicadores['ErrVol_norm']
    Result_Indic = df_indicadores.sort_values(by='Score',ascending=False)
    return True, dfObj_0, Result_Indic

def PlotAnalogias(nomEst,df_union,df_Obs_previo,variable,par_comp,dfBoxPlot,v_resamp):
    df_Obs_previo["meses_str"] = df_Obs_previo["month"].map(map_meses)
    df_Obs_previo.index = df_Obs_previo.index+1
    df_union["meses_str"] = df_union["month"].map(map_meses)
    df_union.index = df_union.index+7

    box_plot_data = []
    box_plot_labels = []
    for index, row in df_Obs_previo.iterrows():
        mes_i = row['month']
        box_plot_data = box_plot_data + [dfBoxPlot.loc[dfBoxPlot[v_resamp] == mes_i,variable].dropna(),]
        box_plot_labels = box_plot_labels + [row['meses_str'],]

    row_df = pd.DataFrame(columns=df_union.columns)
    row_df.index = row_df.index - 1

    df_union = pd.concat([row_df, df_union], ignore_index=False)
    df_union.loc[6,'Prono'] = df_Obs_previo.loc[6,variable]
    df_union.loc[6,'month'] = df_Obs_previo.loc[6,'month']
    df_union.loc[6,'year'] = df_Obs_previo.loc[6,'year']
    df_union = df_union.sort_index()

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)

    ax.scatter(df_Obs_previo.index, df_Obs_previo[variable],s=40,c='blue')
    ax.plot(df_Obs_previo.index, df_Obs_previo[variable],'-',linewidth=2,color='blue',label='Obs')
    for index, row in par_comp.iterrows():
        yr_sim = row['YrSim']
        #ax.scatter(df_union.meses_ord, df_union[yr_sim],s=50,label=int(yr_sim))
        ax.plot(df_union.index, df_union[yr_sim],'-',linewidth=1,label=int(yr_sim))

            
    ax.plot(df_union.index, df_union['Prono'],'-',color='red',linewidth=2,label='Pronostico')
    ax.set_xticks(df_Obs_previo.index)
    ax.set_xticklabels(df_Obs_previo["meses_str"])
    
    ax.boxplot(box_plot_data,patch_artist=True,labels=box_plot_labels,boxprops={'fill': None})

    plt.axvline(x = 5.5, color='k', linestyle='--', label = 'Pronóstico')

    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
    plt.tick_params(axis='y', labelsize=16)
    plt.tick_params(axis='x', labelsize=16,rotation=20)
    plt.xlabel('Fecha', size=18)
    plt.title(nomEst)
    plt.legend(prop={'size':16},loc=0,ncol=1)
    plt.show()
    plt.close()

def MetodoAnalogia(nomEst,df,var,mes_obj,yr_obj,vent_resamp,ParamMetodo):
    longBusqueda = ParamMetodo['longBusqueda']
    longProno = ParamMetodo['longProno']
    orden = ParamMetodo['orden']
    orde_ascending = ParamMetodo['orde_ascending']
    cantidad = ParamMetodo['cantidad']

    # Compara un años con sus parecidos
    _ , dfObj_0, Result_Indic = CalcIndicXFecha(df,yr_obj,mes_obj,longBusqueda,longProno)
    
    if False:   # Compara Indicadores
        x = 'ErrVol_norm'
        y = 'CoefC_norm'
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(Result_Indic[x], Result_Indic[y],2,label=x+' - '+y)
        # ax.plot(Result_Indic.index, Result_Indic['nivel'],'-',label=yr_sim,linewidth=2)
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()
        plt.close()
    
    # Busca los indicadores para el año seleccionado
    R_Indic_i = Result_Indic[(Result_Indic['YrObs'] == yr_obj) & (Result_Indic['MesObs'] == mes_obj)] 

    # Ordena y filtra los primeros n
    R_Indic_i =R_Indic_i.sort_values(by=orden,ascending=orde_ascending).reset_index()

    # Arma el Df de datos Obs para la fecha seleccionada
    fecha_Obj = df.query("year=="+str(yr_obj)+" and month=="+str(mes_obj)) # Busca la fecha seleccionada
    idx_select = fecha_Obj.index.values[0] + 1                                          # Toma el id de la fecha seleccionada
    idx_fecha_f = idx_select + longProno

    index_i = range(idx_select, idx_fecha_f, 1)
    dfObj = pd.DataFrame(index = index_i,columns=['year','month',var])
    dfObj[vent_resamp] = range(mes_obj+1, mes_obj+1+longProno, 1)
    ### Solo para vent_resamp='month' ####
    dfObj['year'] = yr_obj
    dfObj.loc[dfObj[vent_resamp]  > 12, 'year'] = dfObj.loc[dfObj[vent_resamp]  > 12, 'year'] + 1
    dfObj.loc[dfObj[vent_resamp]  > 12, 'month'] = dfObj.loc[dfObj[vent_resamp]  > 12, vent_resamp] - 12

    df_union = dfObj.copy()
    cols_var = [var,'LogVar_Est']

    n_sim_sin_nan = 0
    par_comp = pd.DataFrame(index=range(0,cantidad,1),columns=R_Indic_i.columns)

    for index, row in R_Indic_i.iterrows():
        yr_sim = row['YrSim']
        # Arma el Df para comparar con el seleccionado.
        fecha_sim = df.query("year=="+str(yr_sim)+" and month=="+str(mes_obj))
        idx_sim = fecha_sim.index.values[0] + 1
        idx_sim_f = idx_sim + longProno
        dfSim = df[idx_sim:idx_sim_f].copy().dropna()

        if len(dfSim) < 3:
            print(yr_sim,' Con Faltantes.')
            continue
        else:
            par_comp.iloc[n_sim_sin_nan] = R_Indic_i.iloc[index]
            dfSim = dfSim[['month',]+ cols_var]
            dfSim = dfSim.rename(columns={cols_var[0]:int(yr_sim),cols_var[1]:str(int(yr_sim))+'_Transf'})
            df_union = df_union.merge(dfSim, on='month')
            n_sim_sin_nan += 1
            if n_sim_sin_nan == 5: break

    # Calculo de los pesos
    par_comp['wi'] = 1/par_comp['RMSE']
    par_comp['wi'] = par_comp['wi']/par_comp['wi'].sum()
    
    # Multiplica por los pesos
    for index, row in par_comp.iterrows():
        yrstr = str(row['YrSim'])+'_Transf'
        df_union[yrstr] = df_union[yrstr] * row['wi']


    # Lista  de pronos transformados a sumar
    list_years_analog = [str(yr)+'_Transf' for yr in par_comp['YrSim'].to_list()]
    df_union['Prono'] = df_union[list_years_analog].sum(axis=1)

    # Invierte transformacion
    df_union['mes_mean'] = [df.loc[df['month'] == mes,'LogVar'].dropna().mean() for mes in df_union['month']]
    df_union['mes_std'] =  [df.loc[df['month'] == mes,'LogVar'].dropna().std() for mes in df_union['month']]

    df_union['Prono'] = df_union['Prono']*df_union['mes_std'] + df_union['mes_mean']
    df_union['Prono'] = np.exp(df_union['Prono'])
    
    dfObj_0 = dfObj_0[['year','month',var]].copy()
    df_Obs_previo = df_union[['year','month',var]].copy()
    frames = [dfObj_0, df_Obs_previo]	
    df_Obs_previo = pd.concat(frames).reset_index(drop=True)
    
    #df_union["meses_ord"] = df_union["month"] + 6 - mes_selct
    #df_union.loc[df_union["meses_ord"]  > 12, 'meses_ord'] = df_union.loc[df_union["meses_ord"]  > 12, 'meses_ord'] - 12
    #PlotAnalogias(nomEst,df_union,df_Obs_previo,var,par_comp,df,vent_resamp)

    return df_union

def MetodoAnalogia_errores(name_Est,df,var,vent_resamp,ParamMetodo,
                           CalculaIndicadores=True,connBBDD=None):
    longBusqueda = ParamMetodo['longBusqueda']
    longProno = ParamMetodo['longProno']
    orden = ParamMetodo['orden']
    orde_ascending = ParamMetodo['orde_ascending']
    cantidad = ParamMetodo['cantidad']
    ruta_salidas = ParamMetodo['ruta_salidas']

    if connBBDD != None:
        NombreTabla = 'Salidas_Analog'
        cur = connBBDD.cursor()
        cur.execute('DROP TABLE IF EXISTS '+NombreTabla+';')

    # Indicadores
    if CalculaIndicadores:
        Result_Indic = pd.DataFrame()
        # Quita los ultimos "longBusqueda" y los primeros "longProno" registros. Porque no van a tener la serie completa.
        df_clip = df[longBusqueda:-longProno]
        for index, row in df_clip.iterrows():   # Loop sobre todos los meses desde el inicio de la serie
            mes_selct = int(row['month'])
            yr_select = int(row['year'])
            #if yr_select < 2009: continue
            
            if mes_selct==1: print(yr_select)

            # Compara un año con sus parecidos
            sinNAN, _ , df_indicadores = CalcIndicXFecha(df,yr_select,mes_selct,longBusqueda,longProno)
            if sinNAN:
                Result_Indic = pd.concat([Result_Indic,df_indicadores])

        #print(Result_Indic.head())
        Result_Indic.to_csv(ruta_salidas+'/Indicadores/'+name_Est+'_Indic_Analogias.csv',index=False)


    Result_Indic = pd.read_csv(ruta_salidas+'/Indicadores/'+name_Est+'_Indic_Analogias.csv')
    
    if False:   # Compara Indicadores
        x = 'ErrVol_norm'
        y = 'CoefC_norm'
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.scatter(Result_Indic[x], Result_Indic[y],2,label=x+' - '+y)
        # ax.plot(Result_Indic.index, Result_Indic['nivel'],'-',label=yr_sim,linewidth=2)
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()
        plt.close()
    
    # Calcula error
    columnas = ['YrObs','MesObs','YrSim','nobs', 'Vobs_media', 'Vsim_media', 'Nash', 'CoefC','RMSE', 'SPEDS', 'ErrVol']
    df_indic_error = pd.DataFrame(columns=columnas)

    df_errorXMes = pd.DataFrame(columns=['1er Mes','2do Mes','3er Mes'])

    # Loop sobre todos los meses desde el inicio de la serie. Saca los primeros 10 años para que los primeros años tengan datos para buscar analogia
    for index, row in df_clip.iterrows():
        mes_selct = int(row['month'])
        yr_select = int(row['year'])
        
        if mes_selct==1: print(yr_select)

        # Busca los indicadores para el año seleccionado
        R_Indic_i = Result_Indic[(Result_Indic['YrObs'] == yr_select) & (Result_Indic['MesObs'] == mes_selct)].copy()
        if len(R_Indic_i) < cantidad: continue

        # Falta armar. Para que no calcule dos veces el mismo par
        #R_Indic_i = Result_Indic[((Result_Indic['YrObs'] == yr_select) | (Result_Indic['YrSim'] == yr_select)) & (Result_Indic['MesObs'] == mes_selct)].copy()
        # Ordena y filtra los primeros n
        R_Indic_i =R_Indic_i.sort_values(by=orden,ascending=orde_ascending).reset_index()
        R_Indic_i['YrObs'] = R_Indic_i['YrObs'].astype('int')
        R_Indic_i['YrSim'] = R_Indic_i['YrSim'].astype('int')

        # Arma el Df de datos Obs para la fecha seleccionada
        fecha_Obj = df.query("year=="+str(yr_select)+" and month=="+str(mes_selct))     # Busca la fecha seleccionada
        idx_select = fecha_Obj.index.values[0] + 1                                      # Toma el id de la fecha seleccionada
        idx_fecha_f = idx_select + longProno
        dfObj = df[idx_select:idx_fecha_f].copy()
        del dfObj['Count']

        # Si hay faltantes saltea la serie
        if dfObj[var].isna().sum() > 0: continue

        df_union = dfObj.copy()
        cols_var = [var,'LogVar_Est']
        n_sim_sin_nan = 0
        par_comp = pd.DataFrame(index=range(0,cantidad,1),columns=R_Indic_i.columns)

        for index, row in R_Indic_i.iterrows():
            yr_sim = int(row['YrSim'])

            # Arma el Df para comparar con el seleccionado.
            fecha_sim = df.query("year=="+str(yr_sim)+" and month=="+str(mes_selct))
            idx_sim = fecha_sim.index.values[0] + 1
            idx_sim_f = idx_sim + longProno
            dfSim = df[idx_sim:idx_sim_f].copy().dropna()
            if len(dfSim) < 3:
                print(yr_sim,' Con Faltantes.')
                continue
            else:
                par_comp.iloc[n_sim_sin_nan] = R_Indic_i.iloc[index]
                dfSim = dfSim[['month',]+ cols_var]
                dfSim = dfSim.rename(columns={cols_var[0]:int(yr_sim),cols_var[1]:str(int(yr_sim))+'_Transf'})
                df_union = df_union.merge(dfSim, on='month')
                n_sim_sin_nan += 1
                if n_sim_sin_nan == 5: break
        
        # Calculo de los pesos
        par_comp['wi'] = 1/par_comp['RMSE']
        par_comp['wi'] = par_comp['wi']/par_comp['wi'].sum()

        par_comp['YrObs'] = par_comp['YrObs'].astype('int')
        par_comp['YrSim'] = par_comp['YrSim'].astype('int')
        
        # Multiplica por los pesos
        for index, row in par_comp.iterrows():
            yrstr = str(row['YrSim'])+'_Transf'
            df_union[yrstr] = df_union[yrstr] * row['wi']
        
        list_years_analog = [str(yr)+'_Transf' for yr in par_comp['YrSim'].to_list()]
        df_union['Prono'] = df_union[list_years_analog].sum(axis=1)

        df_union['mes_mean'] = [df.loc[df['month'] == mes,'LogVar'].dropna().mean() for mes in df_union['month']]
        df_union['mes_std'] =  [df.loc[df['month'] == mes,'LogVar'].dropna().std() for mes in df_union['month']]

        df_union['Prono'] = df_union['Prono']*df_union['mes_std'] + df_union['mes_mean']
        df_union['Prono'] = np.exp(df_union['Prono'])

        df_union['Dif_Prono'] = df_union['Prono'] - df_union[cols_var[0]]

        if connBBDD != None:
                df_union['nombre'] = name_Est
                df_union['mes_ant'] = df_union.index + 1
                df_union = df_union[['nombre','year','month','mes_ant',cols_var[0],'Prono','Dif_Prono']]
                df_union.to_sql(NombreTabla, con = connBBDD, if_exists='append',index=False)    # Guarda en BBDD
                connBBDD.commit()

        df_indic_i = IndicadoresDeAjuste(df_union,cols_var[0],'Prono',mes_selct,n_var_obs=yr_select)
        df_indic_error = pd.concat([df_indic_error,df_indic_i])
        df_errorXMes.loc[len(df_errorXMes)] = [df_union.loc[0,'Dif_Prono'],df_union.loc[1,'Dif_Prono'],df_union.loc[2,'Dif_Prono']]


    df_indic_error.to_csv(ruta_salidas+'/IndicadoresAjuste/'+name_Est+'_Indic_Ajuste_Analogia.csv',index=False)
    df_errorXMes.to_csv(ruta_salidas+'/Errores/'+name_Est+'_Error_X_anticipo.csv',index=False)

def TransfDatos(df,var,v_temp,PlotTransf=False):
    #### 1 - Transfroma los datos
    # create log-transformed data
    df['LogVar'] = np.log(df[var])  # Con var < 0 tira warning
    # Normaliza los datos transformados
    df['LogVar_Est'] = np.nan
    for mes in df['month'].unique():
        mes_mean = df.loc[df[v_temp] == mes,'LogVar'].dropna().mean()
        mes_std = df.loc[df[v_temp] == mes,'LogVar'].dropna().std()
        # Normaliza los datos
        df.loc[df['month']==mes,'LogVar_Est'] = (df.loc[df['month']==mes,'LogVar'] - mes_mean)/mes_std
    if PlotTransf:   # plots datos transformados
        fig, axs = plt.subplots(nrows=1, ncols=3)
        #create histograms
        axs[0].hist(df[var], edgecolor='black')
        axs[1].hist(df['LogVar'], edgecolor='black')
        axs[2].hist(df['LogVar_Est'], edgecolor='black')
        #add title to each histogram
        axs[0].set_title('Original Data')
        axs[1].set_title('Log-Transformed Data')
        axs[2].set_title('Log-Transformed-Norm Data')
        plt.show()
        plt.close()

def CalcIndic_Analog_error(df,year_obj,mes_obj,longBusqueda,longProno):
    # True or False. False: si la serie objetivo tiene faltatnes  
    # df_indicadores: Df con los indicadores
    # dfObj_0: Df con La serie objetivo, serie a comprar con resto
    columnas = ['YrObs','MesObs','YrSim','nobs', 'Vobs_media', 'Vsim_media', 'Nash', 'CoefC','RMSE', 'SPEDS', 'ErrVol']
    df_indicadores = pd.DataFrame(columns=columnas)

    variable_transf = 'LogVar_Est'
    # Busca la fecha seleccionada
    fecha_Obj = df.query("year=="+str(year_obj)+" and month=="+str(mes_obj))
    # Toma el id de la fecha seleccionada
    idx_select = fecha_Obj.index.values[0]

    # Arma el Df de datos Obs para la fecha seleccionada
    idx_fecha_fin = idx_select+1
    idx_fecha_inicio = idx_fecha_fin-longBusqueda
    dfObj_0 = df[idx_fecha_inicio:idx_fecha_fin].copy()

    if dfObj_0[variable_transf].isna().sum() > 0:
        return True, 0, dfObj_0
    
    # Compara un año con sus parecidos
    for yr_compara in df['year'].unique():   # Loop sobre todos los meses desde el inicio de la serie
        yr_compara = int(yr_compara)

        if yr_compara == year_obj: continue

        # Arma el Df para comparar con el seleccionado.
        fecha_sim = df.query("year=="+str(yr_compara)+" and month=="+str(mes_obj))
        if len(fecha_sim) == 0:continue
        idx_sim = fecha_sim.index.values[0]
        idx_sim_fin = idx_sim+1
        idx_sim_inicio = idx_sim_fin-longBusqueda

        dfSim = df[idx_sim_inicio:idx_sim_fin].copy()
        dfSim = dfSim[['month',variable_transf]]
        dfSim = dfSim.rename(columns={variable_transf:yr_compara})
        df_union = dfObj_0.merge(dfSim, on='month')

        # Si hay faltantes no calcula los indicadores
        if df_union[yr_compara].isna().sum() > 0: continue
        if len(df_union) == 0: continue

        df_indic_i = IndicadoresDeAjuste(df_union,variable_transf,yr_compara,mes_obj,n_var_obs=year_obj)
        df_indicadores = pd.concat([df_indicadores,df_indic_i])

    return False, df_indicadores, dfObj_0

def MetodoAnalogia_errores_v2(name_Est,df,var,vent_resamp,ParamMetodo,connBBDD=None):
    longBusqueda = ParamMetodo['longBusqueda']
    longProno = ParamMetodo['longProno']
    orden = ParamMetodo['orden']
    orde_ascending = ParamMetodo['orde_ascending']
    cantidad = ParamMetodo['cantidad']
    ruta_salidas = ParamMetodo['ruta_salidas']

    if connBBDD != None:
        NombreTabla = 'Salidas_Analog'
        cur = connBBDD.cursor()
        cur.execute('DROP TABLE IF EXISTS '+NombreTabla+';')
    
    # Calcula error
    columnas = ['nombre','year','month','mes_ant','Caudal','Prono','Dif_Prono','E1','E2','E3','E4','E5']
    df_errorXMes = pd.DataFrame(columns=columnas)
    
    # Quita los ultimos 10 años y los primeros "longProno" registros. Porque no van a tener la serie completa.
    df_clip = df[10*12:-longProno].copy()

    # Loop sobre todos los meses desde el inicio de la serie. Saca los primeros 10 años para que los primeros años tengan datos para buscar analogia
    for index, row in df_clip.iterrows():
        # Si hay valores negativos en la serie se le suma el valor minimo.
        # Luego se le vuelve a sumar el valor. Se hace para que el log no tiere error
        min_val = df[var].min()
        if min_val<=0:
            df[var]=df[var]-(min_val-0.01)

        mes_selct = int(row['month'])
        yr_select = int(row['year'])
        # if yr_select<1944: continue
        # if mes_selct<9: continue
        # print(mes_selct)
        if mes_selct==1: print(yr_select)


        fecha_a_prono = df.query("year=="+str(yr_select)+" and month=="+str(mes_selct))     # Busca la fecha seleccionada
        idx_select = fecha_a_prono.index.values[0]                                          # Toma el id de la fecha seleccionada

        # Filtra datos Futuros
        df_pasado = df[:idx_select+1].copy()

        # Transfroma los datos
        # create log-transformed data
        df_pasado['LogVar'] = np.log(df_pasado[var])

        # Normaliza los datos transformados
        df_pasado['LogVar_Est'] = np.nan

        for mes in df_pasado['month'].unique():
            mes_mean = df_pasado.loc[df_pasado[vent_resamp] == mes,'LogVar'].dropna().mean()
            mes_std = df_pasado.loc[df_pasado[vent_resamp] == mes,'LogVar'].dropna().std()

            # Normaliza los datos
            df_pasado.loc[df_pasado['month']==mes,'LogVar_Est'] = (df_pasado.loc[df_pasado['month']==mes,'LogVar'] - mes_mean)/mes_std
        
        # Calclula Indicadores
        conNAN, df_indicadores, dfObj_0 = CalcIndic_Analog_error(df_pasado,yr_select,mes_selct,longBusqueda,longProno)
        if conNAN: 
            print('Datos Faltantes: ',mes_selct,yr_select)
            continue
        
        # Ordena y filtra los primeros n
        df_indicadores =df_indicadores.sort_values(by=orden,ascending=orde_ascending).reset_index()
        df_indicadores['YrObs'] = df_indicadores['YrObs'].astype('int')
        df_indicadores['YrSim'] = df_indicadores['YrSim'].astype('int')

        # Arma el Df de datos Obs para la fecha seleccionada
        fecha_Obj = df.query("year=="+str(yr_select)+" and month=="+str(mes_selct))     # Busca la fecha seleccionada
        idx_select = fecha_Obj.index.values[0] + 1                                      # Toma el id de la fecha seleccionada
        idx_fecha_f = idx_select + longProno
        dfObj = df[idx_select:idx_fecha_f].copy()
        del dfObj['Count']

        #print(dfObj_0)
        #print(dfObj)

        df_union = dfObj.copy()
        cols_var = [var,'LogVar_Est']
        n_sim_sin_nan = 0
        par_comp = pd.DataFrame(index=range(0,cantidad,1),columns=df_indicadores.columns)
        for index, row in df_indicadores.iterrows():
            yr_sim = int(row['YrSim'])
            # Arma el Df para comparar con el seleccionado.
            fecha_sim = df_pasado.query("year=="+str(yr_sim)+" and month=="+str(mes_selct))
            idx_sim = fecha_sim.index.values[0] + 1
            idx_sim_f = idx_sim + longProno
            dfSim = df_pasado[idx_sim:idx_sim_f].copy().dropna()
            if len(dfSim) < 3:
                print('\t',yr_sim,' con Faltantes.')
                continue
            else:
                par_comp.iloc[n_sim_sin_nan] = df_indicadores.iloc[index]
                dfSim = dfSim[['month',]+ cols_var]
                dfSim = dfSim.rename(columns={cols_var[0]:int(yr_sim),cols_var[1]:str(int(yr_sim))+'_Transf'})
                df_union = df_union.merge(dfSim, on='month')
                n_sim_sin_nan += 1
                if n_sim_sin_nan == 5: break
        
        # Calculo de los pesos
        par_comp['wi'] = 1/par_comp['RMSE']
        par_comp['wi'] = par_comp['wi']/par_comp['wi'].sum()

        par_comp['YrObs'] = par_comp['YrObs'].astype('int')
        par_comp['YrSim'] = par_comp['YrSim'].astype('int')
        
        # Multiplica por los pesos
        for index, row in par_comp.iterrows():
            yrstr = str(row['YrSim'])+'_Transf'
            df_union[yrstr] = df_union[yrstr] * row['wi']
        
        list_years_analog = [str(yr)+'_Transf' for yr in par_comp['YrSim'].to_list()]
        df_union['Prono'] = df_union[list_years_analog].sum(axis=1)

        df_union['mes_mean'] = [df_pasado.loc[df_pasado['month'] == mes,'LogVar'].dropna().mean() for mes in df_union['month']]
        df_union['mes_std'] =  [df_pasado.loc[df_pasado['month'] == mes,'LogVar'].dropna().std() for mes in df_union['month']]

        df_union['Prono'] = df_union['Prono']*df_union['mes_std'] + df_union['mes_mean']
        df_union['Prono'] = np.exp(df_union['Prono'])

        df_union['Dif_Prono'] = df_union['Prono'] - df_union[cols_var[0]]

        df_union['nombre'] = name_Est
        df_union['mes_ant'] = df_union.index + 1

        # Agrega los ensambles para guardarlos
        lst_ensam = par_comp['YrSim'].to_list()
        i = 0
        Ensam_columns = []
        for ensam_i  in lst_ensam:
            i +=1
            df_union = df_union.rename(columns={ensam_i: 'E'+str(i)})
            Ensam_columns += ['E'+str(i),]

        columns_save = ['nombre','year','month','mes_ant',cols_var[0],'Prono','Dif_Prono']+ Ensam_columns
        df_union = df_union[columns_save]

        if min_val<0:
            df_union[var]=df[var]+min_val
            for Ei in Ensam_columns:
                df_union[Ei]  = df_union[Ei]+(min_val-0.01)
        if connBBDD != None:
                df_union.to_sql(NombreTabla, con = connBBDD, if_exists='append',index=False)    # Guarda en BBDD
                connBBDD.commit()
        
        df_errorXMes = pd.concat([df_errorXMes,df_union])
    df_errorXMes.to_csv(ruta_salidas+'/Errores/'+name_Est+'_Error_X_anticipo.csv',index=False)


import scipy.stats

#Cómputo de Error Porcentual Absoluto Medio. 
# Toma por entrada el valor observado (flotante) y un ensamble de pronóstico (lista o array 1D). 
# Computa el error porcentual absoluto de cada miembro de ensamble en relación al valor observado (ape). 
# Finalmente computa su media aritmética (mape). Es un indicador del error de volumen.     
def mape(obs,ensembles=[]):
    ensembles=np.array(ensembles,dtype='float')
    obs=np.array([obs]*len(ensembles),dtype='float')
    ape=abs((ensembles-obs)/obs)*100
    mape=sum(ape)/len(ape)
    return(mape)

# Old School way
# def mape(obs,ensembles=[]):
#     mape=0
#     for member in ensembles:
#         mape=mape+abs((member-obs)/obs)*100
#     return(mape/len(ensembles))

#Cómputo de Rango Percentil. Toma por entrada el valor observado (flotante) y un ensamble de pronóstico (lista o array 1D). 
# Computa la posición del valor observado en la lista de miembros de ensamble. 
# Computa el valor de frecuencia (acumulada) en la distribución empírica observada en el ensamble. 
# Es un indicador de la 'calidad' del ensamble (cuán hábil es para simular la variabilidad observable), esto es: si el ensamble subestima la variabilidad o si la sobreestima. 
# Puede interpretarse como un indicador de tendencia (sesgo). El paquete scipy ofrece una alternativa similar.
def prs(obs,ensembles=[]):
    prs=scipy.stats.percentileofscore(np.array(ensembles),obs)
    return(prs)

# Old School way
# def prs(obs,ensembles=[]):
#     rank=0
#     for member in ensembles:
#         if(member < obs):
#             rank=rank+1
#     return(rank/len(ensembles))

#Cómputo del coeficiente de asociación no paramétrico 'Tau' de Kendall. 
# Se adopta la función del paquete scipy, para cualquier consulta de documentación. 
# Las entradas están constituídas por una lista de valores observados y otra lista de valores simulados. 
# Es un indicador robusto del grado de asociación lineal (dependencia lineal por proporcionalidad o aditividad), 
# con dominio [-1,1] (los valores límites indican asociación perfecta negativa o asociación perfecta positiva, dependencia lineal absoluta, por otro lado 0 indica indepencia lineal).

def tauKendall(obs=[],sim=[]):
    tau=scipy.stats.kendalltau(sim,obs)[0]
    return(tau)

#Cómputo del coeficiente de skill Nash-Sutcliffe. Indicador de la asociación 1:1 entre valores simulados 
# y observados. 
def nashSutcliffeScore(obs=[],sim=[]):
    obs=np.array(obs)
    sim=np.array(sim)
    ns=1-sum((obs-sim)**2)/sum((obs-np.mean(obs))**2)
    return(ns)
