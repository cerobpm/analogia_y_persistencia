# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import datetime as dt
from datetime import timedelta

import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3

from Funciones.FuncSeasonalProno import mape, prs, tauKendall, nashSutcliffeScore

Persistencia = True

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
    var = Estaciones[CB_i]['variable']

    print('\n',nomEst, '       --------------------------')
    nomBBDD = nomEst.strip()
    conn = sqlite3.connect(ruta_salidas+'BD_PronoEstacional_'+nomBBDD+'.sqlite')

    print('\n Analogia')

    sql_query = ('''SELECT * FROM Salidas_Analog''')  
    # WHERE timestart BETWEEN %s AND %s AND unid=%s ORDER BY fecha
    df_Analog = pd.read_sql(sql_query, conn)
    df_Analog = df_Analog.dropna()
    df_Analog = df_Analog.rename(columns={'Caudal':'Obs','Nivel':'Obs','Prono':'Sim'}).dropna()

    print('Nash Sutcliffe Score')
    for mesant in df_Analog['mes_ant'].unique():
        df_clip = df_Analog[df_Analog['mes_ant'] == mesant]
        obs = df_clip['Obs']
        sim = df_clip['Sim']
        
        coef_NshSut = nashSutcliffeScore(obs,sim)
        print (mesant,' - ',round(coef_NshSut,3))

    print('Tau de Kendall')
    for mesant in df_Analog['mes_ant'].unique():
        df_clip = df_Analog[df_Analog['mes_ant'] == mesant]
        obs = df_clip['Obs']
        sim = df_clip['Sim']
        tau_Kendall = tauKendall(obs,sim)
        print (mesant,' - ',round(tau_Kendall,3))


    df_Analog['mape'] = np.nan
    df_Analog['prs'] = np.nan
    for index, row in df_Analog.iterrows():
        obs = row['Obs']
        Ensambles = []
        for Ei in ['E1','E2','E3','E4','E5']:
            Ensambles += [row[Ei],]
        mape_i = mape(obs,ensembles=Ensambles)
        prs_i = prs(obs,ensembles=Ensambles)

        df_Analog.loc[index,'mape'] = mape_i
        df_Analog.loc[index,'prs'] = prs_i

    print('Error Porcentual Absoluto Medio')
    box_plot_data = [df_Analog.loc[df_Analog['mes_ant'] == ant_i,'mape'].dropna() for ant_i in np.sort(df_Analog['mes_ant'].unique())]
    box_plot_labels = [ '1','2','3']
    
    # Grafico
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    #sns.boxplot(data=df_mensual, x="month", y="Caudal",color="skyblue")
    ax.boxplot(box_plot_data,patch_artist=True,labels=box_plot_labels,boxprops={'fill': None})
    plt.title(nomEst+' Error Porcentual Absoluto Medio')    
    plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14,rotation=20)
    plt.xlabel('Anticipo', size=18)
    plt.ylim(0,120)
    #plt.ylabel(label_text, size=18)
    #plt.legend(prop={'size':16},loc=0,ncol=1)
    fname = ruta_salidas+'Figuras/'+nomEst+'_mape_Analogia.png'
    plt.savefig(fname, dpi=150, facecolor='w', edgecolor='w', format='png',bbox_inches = 'tight')
    #plt.show()
    plt.close()

    print('Cómputo de Rango Percentil')
    box_plot_data = [df_Analog.loc[df_Analog['mes_ant'] == ant_i,'prs'].dropna() for ant_i in np.sort(df_Analog['mes_ant'].unique())]
    box_plot_labels = [ '1','2','3']
    
    # Grafico
    fig = plt.figure(figsize=(15, 8))
    ax = fig.add_subplot(1, 1, 1)
    
    #sns.boxplot(data=df_mensual, x="month", y="Caudal",color="skyblue")
    ax.boxplot(box_plot_data,patch_artist=True,labels=box_plot_labels,boxprops={'fill': None})
    plt.title(nomEst+' Cómputo de Rango Percentil')    
    plt.grid(True,axis='y', which='both', color='0.75', linestyle='-.',linewidth=0.3)
    plt.tick_params(axis='y', labelsize=14)
    plt.tick_params(axis='x', labelsize=14,rotation=20)
    plt.xlabel('Anticipo', size=18)
    #plt.ylabel(label_text, size=18)
    #plt.legend(prop={'size':16},loc=0,ncol=1)
    fname = ruta_salidas+'Figuras/'+nomEst+'_RangoPercentil_Analogia.png'
    plt.savefig(fname, dpi=150, facecolor='w', edgecolor='w', format='png',bbox_inches = 'tight')
    #plt.show()
    plt.close()

    print('BoxPlot Errores')
    box_plot_data = [df_Analog[df_Analog['mes_ant'] == mesant]['Dif_Prono'] for mesant in df_Analog['mes_ant'].unique()]
    box_plot_labels = [mesant for mesant in df_Analog['mes_ant'].unique()]

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
    fname = ruta_salidas+'Figuras/'+nomEst+'_BoxPlotErrores_Analogia.png'
    plt.savefig(fname, dpi=150, facecolor='w', edgecolor='w', format='png',bbox_inches = 'tight')
    #plt.show()
    plt.close()