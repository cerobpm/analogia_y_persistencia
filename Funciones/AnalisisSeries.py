# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from datetime import timedelta, datetime, date
from scipy.optimize import fsolve
import requests

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

## Calcula Caudales con la HQ generada por PHC
def curvaHQParana(h):
    return 2.7392 * pow(h,5) - 14.527 * pow(h,4) - 6.2064 * pow(h,3) + 390.16 * pow(h, 2) + 2108.3 * h + 8794.2

# Estadistica Descriptiva
def descript_estadisticos(df,variable):
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

# Variables Temporales
def CreaVariablesTemporales(df):
    df.insert(0, 'year', df.index.year)
    df.insert(1, 'month', df.index.month)
    df.insert(2, 'day', df.index.day)
    df.insert(3, 'yrDay', df.index.dayofyear)
    df.insert(4, 'wkDay', df.index.isocalendar().week)
    #print(df.head(2))

def CreaDFMaxAnual(df,variable):
    #Crea una nueva tabla con los maximos anuales
    df_maxAnual = df[['year',variable]].groupby(['year']).max()
    print(df_maxAnual.head(2))
    return df_maxAnual

## Plots
def Plotea(dfplot,variables,labels,nombres):
    # Grafico de Niveles
    for i,var in enumerate(variables):
        fig = plt.figure(figsize=(15, 8))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(dfplot.index, dfplot[var],'-',label=nombres[i],linewidth=2)
        plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
        plt.tick_params(axis='both', labelsize=16)
        plt.xlabel('Fecha', size=18)
        plt.ylabel(labels[i], size=18)
        plt.legend(prop={'size':16},loc=2,ncol=2 )
        plt.show()
        plt.close()

def PloteaXano(df,var_x,var_y,var_hue):
    sns.lineplot(data=df, x=var_x, y=var_y, hue=var_hue)

    plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14)
    plt.xlabel('Día del Año', size=18)
    if var_y == 'Caudal': label_i = 'Caudal [m'+r'$^3$'+'/s]'
    elif var_y == 'Nivel': label_i = 'Nivel [m]'
    else: label_i = var_y
    plt.ylabel(label_i, size=18)
    plt.show()
    plt.close()

def PlotVarMaxMedMin_Anual(df,variable):
    df_anual = df.groupby(["year"]).agg({ variable: ["max","mean","min"]}).reset_index()
    df_anual.set_index(df_anual['year'], inplace=True)
    del df_anual['year']
    df_anual.columns = ['_'.join(col) for col in df_anual.columns.values]
    df_anual[variable + '_mean'] = df_anual[variable + '_mean'] - df_anual[variable + '_min']
    df_anual[variable + '_max'] = df_anual[variable + '_max'] - df_anual[variable + '_mean']

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    df_anual.plot(kind='bar', stacked=True, color=['skyblue', 'cornflowerblue','darkblue' ], ax=ax)

    plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=9)
    plt.xlabel('Año', size=18)
    
    if variable == 'Caudal': label_i = 'Caudal [m'+r'$^3$'+'/s]'
    elif variable == 'Nivel': label_i = 'Nivel [m]'
    else: label_i = variable

    plt.ylabel(label_i, size=18)
    plt.legend(['Mínimo','Medio','Máximo'],prop={'size':16},loc=0,ncol=1)
    plt.show()
    plt.close()

def PlotVarMaxMedMin_Mensual(df,variable):
    df_mensual = df.groupby(["month"]).agg({ variable: ["max","mean","min"]}).reset_index()
    df_mensual.set_index(df_mensual['month'], inplace=True)
    del df_mensual['month']
    df_mensual.columns = ['_'.join(col) for col in df_mensual.columns.values]
    df_mensual[variable + '_mean'] = df_mensual[variable + '_mean'] - df_mensual[variable + '_min']
    df_mensual[variable + '_max'] = df_mensual[variable + '_max'] - df_mensual[variable + '_mean']

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    df_mensual.plot(kind='bar', stacked=True, color=['skyblue', 'cornflowerblue','darkblue' ], ax=ax)

    plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14,rotation=0)
    plt.xlabel('Mes', size=18)
    
    if variable == 'Caudal': label_i = 'Caudal [m'+r'$^3$'+'/s]'
    elif variable == 'Nivel': label_i = 'Nivel [m]'
    else: label_i = variable

    plt.ylabel(label_i, size=18)
    plt.legend(['Mínimo','Medio','Máximo'],prop={'size':16},loc=0,ncol=1)
    plt.show()
    plt.close()

def PlotMaxAnual(df,variable):
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(df.index, df[variable])
    ax.plot(df.index, df[variable],'-',linewidth=0.5)
    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
    plt.tick_params(axis='both', labelsize=16)
    plt.xlabel('Año', size=18)

    if variable == 'Caudal': label_i = 'Caudal Máximo Anual [m'+r'$^3$'+'/s]'
    elif variable == 'Nivel': label_i = 'Nivel Máximo Anual [m]'
    else: label_i = variable

    plt.ylabel(label_i, size=18)
    #plt.legend(prop={'size':16},loc=2,ncol=2 )
    plt.show()
    plt.close()

def PlotFaltantes(df,variable): # Para ver faltantes
    df[variable] = df[variable].replace(np.nan,-5)

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(df.index, df[variable],'-',linewidth=2)
    plt.grid(True, which='both', color='0.75', linestyle='-.',linewidth=0.5)
    plt.tick_params(axis='both', labelsize=16)
    plt.xlabel('Fecha', size=18)
    #plt.legend(prop={'size':16},loc=2,ncol=2 )
    plt.show()
    plt.close()

    df[variable] = df[variable].replace(-5,np.nan)

# Histograma de Frecuencias
def HistoVariable(df,variable,round_val = 100):
    print('Análisis de frecuencia')
    # round_val se usa para redondear el label de las barras
    from matplotlib.ticker import PercentFormatter

    num_of_bins = round(5*np.log10(len(df))+1)

    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)

    n, bins, patches = ax.hist(df[variable], edgecolor='black', weights=np.ones_like(df[variable])*100 / len(df[variable]), bins=num_of_bins, rwidth=0.9,color='#607c8e')
    #ax.yaxis.set_major_formatter(PercentFormatter())

    bins = [round(item/round_val)*round_val for item in bins]
    plt.xticks(bins)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=12,rotation=45)
    plt.ylabel('Frecuencia de aparición', size=18)

    if variable == 'Caudal': label_i = 'Caudal Máximo Anual [m'+r'$^3$'+'/s]'
    elif variable == 'Nivel': label_i = 'Nivel Máximo Anual [m]'
    else: label_i = variable

    plt.xlabel(label_i, size=18)
    plt.grid(axis='y', alpha=0.75, linewidth=0.3)
    plt.show()

def Permanencia(df,var):
    df_i = df.sort_values(by=var,ascending=False).reset_index(drop=True)
    df_i = df_i[[var,]].dropna()
    df_i['rank'] = df_i.index + 1
    df_i['p_sup'] = df_i['rank'] * 100 / len(df_i)
    print(df_i)

    
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(111)
    ax.plot(df_i.p_sup, df_i[var],'-',linewidth=2)
    plt.tick_params(axis='both', labelsize=16)
    plt.xlabel('Porcentaje de superación [%]', size=18)

    if var == 'Caudal': label_i = 'Caudal [m'+r'$^3$'+'/s]'
    elif var == 'Nivel': label_i = 'Nivel [m]'
    else: label_i = var

    plt.ylabel(label_i, size=18)
    plt.grid(alpha=0.75, linewidth=0.5)
    plt.show()

#

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