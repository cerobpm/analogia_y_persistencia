# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime
from datetime import timedelta, datetime, date
import time

import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

from Funciones.ClasesFunciones import Obj_Serie_v2, prono2serie, GuardaEnBBDD
from Funciones.FuncSeasonalProno import MetodoPersistencia,ErrorXPersistencia
from Funciones.FuncSeasonalProno import TransfDatos, MetodoAnalogia, MetodoAnalogia_errores_v2

# Ultimo mes a partir del cual se hace el pronostico. 
# Se cuenta con datos para este mes. Por lo menos 25 datos diarios.
fecha_emision = datetime.now()

if fecha_emision.day > 27:
    mes_select = fecha_emision.month
    yr_select = fecha_emision.year
else:
    if fecha_emision.month == 1:
        mes_select = 12
        yr_select = fecha_emision.year-1
    else:
        mes_select = fecha_emision.month-1
        yr_select = fecha_emision.year

#mes_select = 6
#yr_select = 2023
print('Mes: ',mes_select)
print('Año: ',yr_select)

saveBBDD_Alerta = True
Plot = False #   True    False

timestart = "1905-01-01T01:00:00Z"
timeend = "2028-05-10T12:00:00Z"

# Lista de estaciones
Estaciones = {
    'Concepcion': { 'name': 'Concepcion' ,
                    'origen':'BBDD',
                    'id_serie':1004,#155,
                    'series_id_qmm_sim':24657,
                    'series_id_hmm_sim':34832,
                    'frecAdopt':'D',
                    'HQ': False,
                    'variable': 'Caudal',    # Nivel, Caudal
                    'vent_resamp': 'month',
                    'lim_outliers': [0,10000],#-1.5,8.0
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3           #  Cantidad de faltantes consecutivos que completa la interpolacion
                },    
    'Puerto Pilcomayo': {   'name': 'Puerto Pilcomayo' ,
                            'origen':'BBDD',
                            'id_serie':904,# 55 904
                            'series_id_qmm_sim':24658,
                            'series_id_hmm_sim':34833,
                            'frecAdopt':'D',
                            'HQ': False,
                            'variable': 'Caudal',# Nivel Caudal
                            'vent_resamp': 'month',
                            'lim_outliers': [0,40000],# -1.5,8.0]
                            'lim_salto':0.5,
                            'timeHInterval':24,     #  Horas. Un dato por dia
                            'limIterp':3,           #  Cantidad de faltantes consecutivos que completa la interpolacion
                        },
    'Corrientes': { 'name': 'Corrientes' ,
                    'origen':'BBDD',
                    'id_serie':868,# 868 16
                    'series_id_qmm_sim':25051,
                    'series_id_hmm_sim':34834,
                    'frecAdopt':'D',
                    'HQ': False,
                    'variable': 'Caudal',  # Nivel
                    'vent_resamp': 'month',
                    'lim_outliers': [0,50000], # -1.5,8.0
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3            #  Cantidad de faltantes consecutivos que completa la interpolacion
                },
    'Parana': { 'name': 'Parana' ,          # Nombre de la Estacion
                'origen':'BBDD',            # Orgine de los datos BBDD o CSV
                'id_serie':29,
                'series_id_qmm_sim':25080,  # Persistencia Caudal
                'series_id_hmm_sim':34829,  # Persistencia Nivel

                'frecAdopt':'D',
                'HQ': True,                 # True si como dato de la serie se tiene el nivel y se lo quiere convertis en caudal / Se debe tener la ec. de la HC
                'variable': 'Caudal',       # Cual es la variable a pronosticar. Nivel o Caudal
                'vent_resamp': 'month',     # ventana temporal para resamplear:year  month  day  yrDay  wkDay  
                'lim_outliers': [-1.5,8.0],
                'lim_salto':0.5,
                'timeHInterval':24,     #  Horas. Un dato por dia
                'limIterp':3,           #  Cantidad de faltantes consecutivos que completa la interpolacion
                'EstComplementarias':{
                            'SantaFe':{     'Abrev':'h_stFe',
                                            'origen':'BBDD',
                                            'id_serie':30,
                                            'lim_outliers':[-1.5,8.0],
                                            'lim_salto':0.5,
                                            'frecAdopt':'D',
                                            'dtime':-1},
                            'Diamante':{'Abrev':'h_Diam',
                                        'origen':'BBDD',
                                        'id_serie':31,
                                        'lim_outliers':[-1.5,8.0],
                                        'lim_salto':0.5,
                                        'frecAdopt':'D',
                                        'dtime':-1}}
            },
    'El Colorado': {    'name': 'El Colorado' ,
                    'origen':'BBDD',
                    'id_serie':6522,# 25 874
                    'series_id_qmm_sim':25116,
                    'series_id_hmm_sim':34835,
                    'frecAdopt':'D',
                    'HQ': False,
                    'variable': 'Caudal',# Caudal    Nivel
                    'vent_resamp': 'month',
                    'lim_outliers': [0,80000],# -1.5,8.0
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3,           #  Cantidad de faltantes consecutivos que completa la interpolacion
                },    
    'Andresito': {  'name': 'Andresito' ,
                    'origen':'BBDD',
                    'id_serie':26605,# 8 , 857
                    'series_id_qmm_sim':25060,
                    'series_id_hmm_sim':34831,
                    'frecAdopt':'D',
                    'HQ': False,
                    'variable': 'Caudal',#Nivel  Caudal
                    'vent_resamp': 'month',
                    'lim_outliers': [0,40000],# [-1.5,8.0],
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3          #  Cantidad de faltantes consecutivos que completa la interpolacion
                },}


Estaciones = { 
    'Puerto Pilcomayo': {   'name': 'Puerto Pilcomayo' ,
                            'origen':'BBDD',
                            'id_serie':904,# 55 904
                            'series_id_qmm_sim':24658,
                            'series_id_hmm_sim':34833,
                            'frecAdopt':'D',
                            'HQ': False,
                            'variable': 'Caudal',# Nivel Caudal
                            'vent_resamp': 'month',
                            'lim_outliers': [0,40000],# -1.5,8.0]
                            'lim_salto':0.5,
                            'timeHInterval':24,     #  Horas. Un dato por dia
                            'limIterp':3,           #  Cantidad de faltantes consecutivos que completa la interpolacion
                        }, 
    'Andresito': {  'name': 'Andresito' ,
                    'origen':'BBDD',
                    'id_serie':26605,# 8 , 857
                    'series_id_qmm_sim':25060,
                    'series_id_hmm_sim':34831,
                    'frecAdopt':'D',
                    'HQ': False,
                    'variable': 'Caudal',#Nivel  Caudal
                    'vent_resamp': 'month',
                    'lim_outliers': [0,40000],# [-1.5,8.0],
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3          #  Cantidad de faltantes consecutivos que completa la interpolacion
                },
    'Itaipu': {    'name': 'Itaipu' ,
        'origen':'BBDD',
        'id_serie':941,# 9691  1790
        'series_id_qmm_sim':25143,#25116,
        'series_id_hmm_sim':34848,#34835,
        'frecAdopt':'D',
        'HQ': False,
        'variable': 'Caudal',# Caudal    Nivel
        'vent_resamp': 'month',
        'lim_outliers': [3000,80000],# -1.5,8.0
        'lim_salto':0.5,
        'timeHInterval':24,     #  Horas. Un dato por dia
        'limIterp':3,           #  Cantidad de faltantes consecutivos que completa la interpolacion
                },}

## Calcula Caudales en Parana con la curva HQ generada por PHC 
def curvaHQParana(h):
    return 2.7392 * pow(h,5) - 14.527 * pow(h,4) - 6.2064 * pow(h,3) + 390.16 * pow(h, 2) + 2108.3 * h + 8794.2

## Parametros de los modelos
Param_Modelos = {
    'filePath': '/Datos/',
    'ruta_salidas': 'Results/',

    'longBusqueda': 6,         # longitud de la serie para buscar Analogas
    'longProno':4,              # longitud del pronostico 
    
    # Analogias
    'orden':'RMSE',         # Ordena Por
    'orde_ascending':True,
    'cantidad':5            # Cantidad de series que toma
    }

dic_Metodo = { 'Persistencia':{ 'Prono':True,
                                'Calc_Error':False}}

series_persist = []

for CB_i in list(Estaciones.keys()):
    inici_0 = time.time()
    nomEst = Estaciones[CB_i]['name']
    #if nomEst !=  'Itaipu':  continue

    Parametros = Estaciones[nomEst]         # Parametros de la serie

    serie_id = int(Parametros['id_serie'])
    vent_resamp = Parametros['vent_resamp'] # mensual

    Parametros['timestart'] = timestart
    Parametros['timeend'] = timeend

    if Parametros['variable'] == 'Nivel':
        serie_id_save = Parametros['series_id_hmm_sim']
    elif  Parametros['variable'] == 'Caudal':
        serie_id_save = Parametros['series_id_qmm_sim']    
    
    nomBBDD = nomEst.strip()
    #conn = sqlite3.connect(Param_Modelos['ruta_salidas']+'BD_PronoEstacional_'+nomBBDD+'.sqlite')

    origen = Parametros['origen']           
    HQ = Parametros['HQ']                   
    
    file_path = Param_Modelos['filePath']   # Ruta a los datos
    limite_outliers = Parametros['lim_outliers']

    # Crea el objeto serie
    # Y consulta las series de la BBDD de Alerta (o csv)
    ObjSerie_i = Obj_Serie_v2(serie_id,Parametros)
    print('\nSerie: ',ObjSerie_i.name)
    print('Serie_id: ',ObjSerie_i.id)
    
    # Elimina outliers
    outliers, ObjSerie_i.serie = ObjSerie_i.removeOutliers(ObjSerie_i.serie,limite_outliers,column=ObjSerie_i.var)
    # Regulariza la serie
    ObjSerie_i.SerieRegulariza(Parametros)  
    df_obs = ObjSerie_i.serie_reg.copy()
    print(ObjSerie_i.serie.index.min())
    print(ObjSerie_i.serie.index.max())
    if Plot:
        ObjSerie_i.PlotFaltantes(ObjSerie_i.serie,ObjSerie_i.var) 
        ObjSerie_i.PlotFaltantes(ObjSerie_i.serie_reg,ObjSerie_i.var)
    print('\nDatos Faltantes:')
    print('NaN: '+str(ObjSerie_i.NNaN))
    
    if HQ == True:
        print('HQ')
        if ObjSerie_i.id == 29:
            ## Calcula Caudales con la HQ generada por PHC
            ObjSerie_i.serie_reg['Caudal'] = ObjSerie_i.serie_reg.apply(lambda row : curvaHQParana(row[ObjSerie_i.var]), axis = 1).round(2)   
            ObjSerie_i.var = 'Caudal'
    
    # Crea columna con el año, el mes, el dia y el dia del año, wkDay
    ObjSerie_i.CreaVariablesTemporales(ObjSerie_i.serie_reg)  
    
    # Resamplea los datos.
    df_resamp = ObjSerie_i.ResampleSerie(ObjSerie_i.serie_reg,vent_resamp,ObjSerie_i.var)          
    df_resamp = df_resamp.rename(columns={'Variable':ObjSerie_i.var})
    #print(df_resamp)

    # Configuraciones de los modelos.
    longBusqueda = Param_Modelos['longBusqueda']
    longProno = Param_Modelos['longProno']

    print('Desde: ',int(df_resamp.loc[0,[vent_resamp,]].values[0]),'/',int(df_resamp.loc[0,['year',]].values[0]))
    print('Hasta: ',int(df_resamp.loc[len(df_resamp)-1,[vent_resamp,]].values[0]),'/',int(df_resamp.loc[len(df_resamp)-1,['year',]].values[0]))

    df_resamp.loc[df_resamp['Count'] < 25, ObjSerie_i.var] = np.nan         # Si el mes tiene menos de 25 datos reemplaza el promedio por un vacio
    print('NaN Mensuales: '+str(df_resamp[ObjSerie_i.var].isna().sum()))    # Cantidad de Faltantes
    df_resamp = df_resamp.round(2)
    #print(df_resamp.tail(2))

    ### Metodo Persistencia.
    if dic_Metodo['Persistencia']['Prono']:  # Arma Figura para un mes en particular
        df_prono_presis = MetodoPersistencia(nomEst,df_resamp,ObjSerie_i.var,mes_select,yr_select,longBusqueda,longProno,vent_resamp,Prono=True)

        # Mes a formato fecha
        def month2Date(x):
                return datetime(yr_select, x, 1,0,0,0,0)
        df_prono_presis['month'] = df_prono_presis['month'].apply(month2Date)

        # Formato para guardar
        series_persist += [prono2serie( df_prono_presis,
                                        main_colname="VarProno",
                                        members={},
                                        series_id=serie_id_save)]
        

    if dic_Metodo['Persistencia']['Calc_Error']: # Calcula Error
        ErrorXPersistencia(nomEst,df_resamp,ObjSerie_i.var,longBusqueda,longProno,vent_resamp,Plot=False,connBBDD=conn)
    
    fin_0 = time.time()
    print('Tiempo : ',round((fin_0-inici_0)/60,2),' minutos')

if saveBBDD_Alerta:
    print('Guarda en BBDD')
    if dic_Metodo['Persistencia']['Prono']: 
        GuardaEnBBDD(series_persist,fecha_emision,493)# Guarda en BBDD




Otras = {'Rosario': {  'name': 'Rosario' ,
                    'origen':'BBDD',
                    'id_serie':26631,# 34, 883
                    'series_id_qmm_sim':25085,
                    'series_id_hmm_sim':34830,
                    'frecAdopt':'H',
                    'HQ': False,
                    'variable': ' Caudal',# Nivel Caudal
                    'vent_resamp': 'month',
                    'lim_outliers': [0,100000],#-1.5,8.0    0,100000
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3          #  Cantidad de faltantes consecutivos que completa la interpolacion
                },


    'El Colorado': {    'name': 'El Colorado' ,
                    'origen':'BBDD',
                    'id_serie':6522,# 25 874
                    'series_id_qmm_sim':25116,
                    'series_id_hmm_sim':34835,
                    'frecAdopt':'D',
                    'HQ': False,
                    'variable': 'Caudal',# Caudal    Nivel
                    'vent_resamp': 'month',
                    'lim_outliers': [0,80000],# -1.5,8.0
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3,           #  Cantidad de faltantes consecutivos que completa la interpolacion
                },   }