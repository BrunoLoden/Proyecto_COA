# -*- coding: utf-8 -*-
"""
Created on Sun Dec 26 21:19:32 2021

@author: Bruno
"""
T =   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]
#========================Escoger año de simulacion (2021 o 2050)===============
anho = 2021
#==============================================================================
if anho == 2021:
    #==========================================================================
    #=====================Datos de la unidad hidraulica 2021===================
    #==========================================================================
    H = ['H1', 'H2', 'H3', 'H4']
    psi =   {'H1':12.3,'H2':9.8, 'H3':15.3, 'H4':25} #factor de turbinamiento
    V_min = {'H1':80 , 'H2':60,  'H3':100,  'H4':70}#Volumen minimo de reservorio
    V_max = {'H1':150, 'H2':120, 'H3':240,  'H4':180}#volumen maximo de reservorio
    V_ini = {'H1':100, 'H2':80,  'H3':240,  'H4':170}#Volumen inicial del reservorio
    V_fin = {'H1':120, 'H2':70,  'H3':170,  'H4':140}#Volumen final del reservorio
    U_min = {'H1':5  , 'H2':6,   'H3':10,   'H4':6}#Caudal de turbinamiento minimo
    U_max = {'H1':15 , 'H2':15,  'H3':30,   'H4':20}#caudal de turbinamiento maximo
    Ph_max ={'H1':500, 'H2':500, 'H3':500,  'H4':500}#Maxima capacidad de generacion
    
    
    
    #Afluencia: Caudal que llega al reservorio de un generador hidraulico "Hi"
    #Afluencia para Generador Hidraulico 1
    Yh1 = {1:15,  2:9,   3:8,   4:7,   5:6,   6:7,   7:8,   8:9, 
    	   9:10, 10:11, 11:12, 12:10, 13:12, 14:11, 15:12, 16:10, 
    	  17:9,  18:8,  19:7,  20:6,  21:7,  22:8,  23:9,  24:10}
    #Afluencia para Generador Hidraulico 2
    Yh2 = {1:15,  2:8,  3:9,  4:9,  5:8,  6:7,  7:6,  8:7, 
    	   9:8, 10:9, 11:9, 12:8, 13:8, 14:9, 15:9, 16:8, 
    	  17:7, 18:6, 19:7, 20:8, 21:9, 22:9, 23:8, 24:8}
    #Afluencia para Generador Hidraulico 3
    Yh3 = {1:5,  2:8,  3:9,  4:9,  5:8,  6:7,  7:6,  8:7, 
    	   9:8, 10:9, 11:9, 12:8, 13:8, 14:9, 15:9, 16:8, 
    	  17:7, 18:6, 19:7, 20:8, 21:9, 22:9, 23:8, 24:8}
    #Afluencia para Generador Hidraulico 4
    Yh4 = {1:8,  2:8,  3:3,  4:9,  5:8,  6:7,  7:6,  8:7, 
    	   9:8, 10:9, 11:9, 12:6, 13:8, 14:9, 15:9, 16:8, 
    	  17:7, 18:6, 19:7, 20:8, 21:9, 22:9, 23:8, 24:8}
    
    
    #Variable "Y" sera el diccionario que idicara la afluencia en cada generador
    #   para cafa hora del dia
    Y = {}
    for t in T:
        for h in H:
            if h == 'H1':
                Y[(h,t)] = Yh1[t]
            if h == 'H2':
                Y[(h,t)] = Yh2[t]
            if h == 'H3':
                Y[(h,t)] = Yh3[t]
            if h == 'H4':
                Y[(h,t)] = Yh4[t]


elif anho == 2050:
    #==========================================================================
    #==================Datos de la unidad hidraulica 2050======================
    #==========================================================================
    H = ['H1', 'H2']
    psi =   {'H1':12.3,'H2':9.8} #factor de turbinamiento
    V_min = {'H1':80 , 'H2':60}#Volumen minimo de reservorio
    V_max = {'H1':150, 'H2':120}#volumen maximo de reservorio
    V_ini = {'H1':100, 'H2':80}#Volumen inicial del reservorio
    V_fin = {'H1':120, 'H2':70}#Volumen final del reservorio
    U_min = {'H1':5  , 'H2':6}#Caudal de turbinamiento minimo
    U_max = {'H1':15 , 'H2':15}#caudal de turbinamiento maximo
    Ph_max ={'H1':500, 'H2':500}#Maxima capacidad de generacion
    
 
    #Afluencia: Caudal que llega al reservorio de un generador hidraulico "Hi"
    #Afluencia para Generador Hidraulico 1
    Yh1 = {1:10,  2:9,   3:8,   4:7,   5:6,   6:7,   7:8,   8:9, 
    	   9:10, 10:11, 11:12, 12:10, 13:12, 14:11, 15:12, 16:10, 
    	  17:9,  18:8,  19:7,  20:6,  21:7,  22:8,  23:9,  24:10}
    #Afluencia para Generador Hidraulico 2
    Yh2 = {1:8,  2:8,  3:9,  4:9,  5:8,  6:7,  7:6,  8:7, 
    	   9:8, 10:9, 11:9, 12:8, 13:8, 14:9, 15:9, 16:8, 
    	  17:7, 18:6, 19:7, 20:8, 21:9, 22:9, 23:8, 24:8}


    #Variable "Y" sera el diccionario que idicara la afluencia en cada generador
    #   para cafa hora del dia
    Y = {}
    for t in T:
        for h in H:
            if h == 'H1':
                Y[(h,t)] = Yh1[t]
            if h == 'H2':
                Y[(h,t)] = Yh2[t]


else:
    print("Error en año - Hidros")