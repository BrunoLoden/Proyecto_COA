# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 21:22:54 2021

@author: Bruno
"""
anho = 2021

if anho == 2021:
    #==========================================================================
    #===========================Datos de Eolico 2021===========================
    #==========================================================================
    E = ['E1', 'E2', 'E3']                     #0
    #Ingresando los parámetros del generador --eolico
    be =     {'E1':4.8,  'E2':6.57, 'E3':5.55} #1
    ce =     {'E1':89,    'E2':83,    'E3':100}#2
    Pe_min = {'E1':28,    'E2':20,    'E3':30 }#3
    Pe_max = {'E1':200,   'E2':290,   'E3':190}#4
    RUe =    {'E1':40,    'E2':30,    'E3':30} #5
    RDe =    {'E1':40,    'E2':30,    'E3':30} #6
    
    #velocidad del viento (para cada hora)
    Wind = {1 :44.1,  2 :48.5,  3 :65.7,  4 :144.9, 5 :202.3, 6 :317.3, 
            7 :364.4, 8 :317.3, 9 :271,   10:306.9, 11:424.1, 12:398, 
            13:487.6, 14:521.9, 15:541.3, 16:560,   17:486.8, 18:372.6,
            19:367.4, 20:314.3, 21:316.6, 22:311.4, 23:405.4, 24:470.4}
    #penalidad
    VWC=50

elif anho == 2050:
    #=============================================================================
    #============================Datos de Eolico 2050=============================
    #=============================================================================
    E = ['E1', 'E2', 'E3', 'E4', 'E5', 'E6', 'E7','E8', 'E9']
    #Ingresando los parámetros del generador --eolico
    be =     {'E1':4.8,   'E2':6.57,  'E3':5.55,  'E4':6.21,  'E5':5.55, 'E6':6.57,  'E7':6.21, 'E8':6.57,  'E9':6.21}
    ce =     {'E1':89,    'E2':83,    'E3':100,   'E4':70,    'E5':100,  'E6':83,    'E7':70,   'E8':83,    'E9':70}
    Pe_min = {'E1':28,    'E2':20,    'E3':30,    'E4':20,    'E5':30,   'E6':20,    'E7':20,   'E8':20,    'E9':20}
    Pe_max = {'E1':200,   'E2':290,   'E3':190,   'E4':260,   'E5':190,  'E6':290,   'E7':260,  'E8':290,   'E9':260}
    RUe =    {'E1':40,    'E2':30,    'E3':30,    'E4':50,    'E5':30,   'E6':30,    'E7':50,   'E8':30,    'E9':50}
    RDe =    {'E1':40,    'E2':30,    'E3':30,    'E4':50,    'E5':30,   'E6':30,    'E7':50,   'E8':30,    'E9':50}
    
    #velocidad del viento (para cada hora)
    Wind = {1 :44.1,  2 :48.5,  3 :65.7,  4 :144.9, 5 :202.3, 6 :317.3, 
            7 :364.4, 8 :317.3, 9 :271,   10:306.9, 11:424.1, 12:398, 
            13:487.6, 14:521.9, 15:541.3, 16:560,   17:486.8, 18:372.6,
            19:367.4, 20:314.3, 21:316.6, 22:311.4, 23:405.4, 24:470.4}
    #penalidad
    VWC=50

else:
    print("Error en año - Eolicas")