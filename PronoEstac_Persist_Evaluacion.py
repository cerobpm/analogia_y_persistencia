# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

from Funciones.FuncSeasonalProno import mape, prs, tauKendall, nashSutcliffeScore

Estaciones = {
    'Parana': { 'name': 'Parana' ,
                'origen':'BBDD',
                'id_serie':29,
                'frecAdopt':'D',
                'HQ': True,
                'variable': 'Caudal',
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
    'Rosario': {  'name': 'Rosario' ,
                    'origen':'BBDD',
                    'id_serie':34,
                    'frecAdopt':'H',
                    'HQ': False,
                    'variable': 'Nivel',
                    'vent_resamp': 'month',
                    'lim_outliers': [-1.5,8.0],
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3          #  Cantidad de faltantes consecutivos que completa la interpolacion
                },
    'Andresito': {  'name': 'Andresito' ,
                    'origen':'BBDD',
                    'id_serie':8,
                    'frecAdopt':'D',
                    'HQ': False,
                    'variable': 'Nivel',
                    'vent_resamp': 'month',
                    'lim_outliers': [-1.5,8.0],
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3          #  Cantidad de faltantes consecutivos que completa la interpolacion
                },
    'Concepcion': { 'name': 'Concepcion' ,
                    'origen':'BBDD',
                    'id_serie':155,
                    'frecAdopt':'D',
                    'HQ': False,
                    'variable': 'Nivel',
                    'vent_resamp': 'month',
                    'lim_outliers': [-1.5,8.0],
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3           #  Cantidad de faltantes consecutivos que completa la interpolacion
                },
    'Puerto Pilcomayo': {   'name': 'Puerto Pilcomayo' ,
                            'origen':'BBDD',
                            'id_serie':55,
                            'frecAdopt':'D',
                            'HQ': False,
                            'variable': 'Nivel',
                            'vent_resamp': 'month',
                            'lim_outliers': [-1.5,8.0],
                            'lim_salto':0.5,
                            'timeHInterval':24,     #  Horas. Un dato por dia
                            'limIterp':3,           #  Cantidad de faltantes consecutivos que completa la interpolacion
                        },
    'Corrientes': { 'name': 'Corrientes' ,
                    'origen':'BBDD',
                    'id_serie':19,
                    'frecAdopt':'D',
                    'HQ': False,
                    'variable': 'Nivel',
                    'vent_resamp': 'month',
                    'lim_outliers': [-1.5,8.0],
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3            #  Cantidad de faltantes consecutivos que completa la interpolacion
                },
    'San Javier': { 'name': 'San Javier' ,
                   'origen':'BBDD',
                    'id_serie':65,
                    'frecAdopt':'D',
                    'HQ': False,
                    'variable': 'Nivel',
                    'vent_resamp': 'month',
                    'lim_outliers': [-1.5,8.0],
                    'lim_salto':0.5,
                    'timeHInterval':24,     #  Horas. Un dato por dia
                    'limIterp':3,           #  Cantidad de faltantes consecutivos que completa la interpolacion
                    
                }}

ruta_salidas = 'Results/'


for CB_i in list(Estaciones.keys()):
    nomEst = Estaciones[CB_i]['name']       # Nombre de la Estacion

    print('\n',nomEst, '       --------------------------')
    nomBBDD = nomEst.strip()
    conn = sqlite3.connect(ruta_salidas+'BD_PronoEstacional_'+nomBBDD+'.sqlite')

    print('\n Persistencia')

    sql_query = ('''SELECT * FROM Salidas_Persist''')  
    # WHERE timestart BETWEEN %s AND %s AND unid=%s ORDER BY fecha
    df_Persist = pd.read_sql(sql_query, conn)

    if 'Caudal_Obs' in df_Persist.columns:
        var = 'Caudal'
    elif 'Nivel_Obs' in df_Persist.columns:
        var = 'Nivel'
    
    df_Persist = df_Persist.rename(columns={'Caudal_Obs':'Obs','Nivel_Obs':'Obs','Caudal_Prono':'Sim','Nivel_Prono':'Sim'})
    
    print('Nash Sutcliffe Score')
    for mesant in df_Persist['mes_ant'].unique():
        df_clip = df_Persist[df_Persist['mes_ant'] == mesant]
        obs = df_clip['Obs']
        sim = df_clip['Sim']
        
        coef_NshSut = nashSutcliffeScore(obs,sim)
        print (mesant,' - ',round(coef_NshSut,3))

        Vobs_media = round(np.mean(obs),1)
        Vsim_media = round(np.mean(sim),1)

        #Nash y Sutcliffe
        F = (np.square(np.subtract(sim, obs))).sum()
        F0 = (np.square(np.subtract(obs, np.mean(obs)))).sum()
        E_var = round(100*(F0-F)/F0,3)

    print('Tau de Kendall')
    for mesant in df_Persist['mes_ant'].unique():
        df_clip = df_Persist[df_Persist['mes_ant'] == mesant]
        obs = df_clip['Obs']
        sim = df_clip['Sim']
        
        tau_Kendall = tauKendall(obs,sim)
        print (mesant,' - ',round(tau_Kendall,3))


    print('BoxPlot Errores')
    box_plot_data = [df_Persist[df_Persist['mes_ant'] == mesant]['Dif_Prono'] for mesant in df_Persist['mes_ant'].unique()]
    box_plot_labels = [mesant for mesant in df_Persist['mes_ant'].unique()]

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
    fname = ruta_salidas+'Figuras/'+nomEst+'_BoxPlotErrores_Persist.png'
    plt.savefig(fname, dpi=150, facecolor='w', edgecolor='w', format='png',bbox_inches = 'tight')
    #plt.show()
    plt.close()