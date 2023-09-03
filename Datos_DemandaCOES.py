# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 22:42:49 2021

@author: Bruno
"""

import pandas as pd

#-----------------------------------------------------------
#--------------------Datos de Excel--------------------------
Nom_archivo ="Demanda_30112021.xlsx"
Nom_Hoja = "Datos" #3 Hojas
#-----------------------------------------------------------
df = pd.read_excel(Nom_archivo,Nom_Hoja)#Extrayendo iformacion de excel en data frame
#----Diccionario de la demanda para los 24 periodos de operacion-------
list_dem = df["Demanda en proporcion"].tolist()
list_per = df["Periodo"].tolist()
zip_dem =zip(list_per,list_dem)
#--Diccionario---
Dem = dict(zip_dem)
# print(Dem)
#-----Proporcion con la demanda real del COES para el dia 30/11/2021-------
list_pro = df["Proporcion 1/n"].tolist()
prop = list_pro[0]

# print(type(list_dem[1]))
# print(type(list_per[1]))