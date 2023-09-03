# -*- coding: utf-8 -*-
# """
# Created on Mon Dec 13 01:13:17 2021

# @author: Bruno
# """
#Modelo uninodal, en un mismo nodo estaran conectadops las generadoras termicas
#   ,hidroelectricas y eolicas.
# import pyomo as pyo
import pyomo.environ as pye
import pandas as pd

Fecha = 2021

from Datos_Eolicas         import E  ,be  ,ce   ,Pe_min ,Pe_max ,RUe   ,RDe  ,Wind ,VWC
from Datos_Hidroelectricas import H  ,psi ,V_min,V_max  ,V_ini  ,V_fin ,U_min,U_max,Ph_max,Y
from Datos_Termicas        import G  ,bg  ,cg   ,Pt_max ,Pt_min ,RUt   ,RDt  ,anho
from Datos_DemandaCOES     import Dem,prop
import SolverCOAn as scoa


de=["E","be","ce","Pe_min","Pe_max","RUe", "RDe","Wind","VWC"]
# Ee=d_e.l_p[0]
# Ei=d_e.l_w[0]






#--------------Datos de la demanda-------------No depende del generador
#Periodos de operacion
T =   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]


prop = int(prop)

#=============================================================================
#=============================================================================
#Definiendo modelo concreto---------------------------------------------------
model = pye.ConcreteModel()
#Definimos funcion para limites de la potencia mini y max de Gen Eolicas------
def bounds_Ge (model,e,t):
    return (Pe_min[e],Pe_max[e])

#Definimos variablede la potencia---------------------------------------------
model.P_Ge = pye.Var(E,T, bounds=bounds_Ge)

#Defiimos potencia ((eolica desperdiciada):pontencia entregada mayor a la necesaria)
model.Pwc = pye.Var(T,domain  = pye.NonNegativeReals)
#Definimos potencia-----------------------------------------------------------
model.Pw = pye.Var(T,domain = pye.NonNegativeReals)

#=============================================================================
#=============================================================================

#Creamos nuestro modelo 

#Definimos limites y variables para generadores termicos e hidraulicos
#definimos limites
def bounds_Gt(model,g,t):
    return (Pt_min[g],Pt_max[g])
model.P_Gt = pye.Var(G,T,bounds=bounds_Gt)

def bounds_Gh(model,h,t):
    return (0,Ph_max[h])
model.P_Gh = pye.Var(H,T,bounds=bounds_Gh)
#---Volumen en reservorio------------------------
def bounds_Vol(model,h,t):
    return (V_min[h],V_max[h])
model.Vol = pye.Var(H,T, bounds =  bounds_Vol)
#---Caudal de turbinado--------------------------
def bounds_cau_turb(model, h, t):
    return (U_min[h],U_max[h])
model.Cau_turb = pye.Var(H,T, bounds = bounds_cau_turb)
#---Caudal de Vertimiento------------------------
model.Vert = pye.Var(H,T, domain = pye.NonNegativeReals)

for h in H:
    model.Vol[h,1].fix(V_ini[h])
    model.Vol[h,24].fix(V_fin[h])


#=============================================================================
#====================Definimos funcion objetivo (lineal)======================
model.obj_t = pye.Objective(expr = \
                    sum(model.P_Gt[g,t]*bg[g]+cg[g] for g in G for t in T)+\
                    sum(model.P_Ge[e,t]*be[e]+ce[e]+VWC*model.Pwc[t] for e in E for t in T))
                    
#=============================================================================
#=============================================================================

#----------------Definimos restricciones-------------
#Balance, potencia producido por ambos debe de ser igual a la demanda
#Balance de energia electrica

def Balance_energia(model,t):
    return model.Pw[t] + sum(model.P_Ge[e,t] for e in E)\
                       + sum(model.P_Gt[g,t] for g in G)\
                       + sum(model.P_Gh[h,t] for h in H) == Dem[t]
model.Balance = pye.Constraint(T, rule = Balance_energia)

#=============================================================================
#==============================Restricciones Hidroelectrica===================
#=============================================================================
#Definimos restriccion de generacion hidraulica
def regla_gen_hidro(model,h,t):
    return model.P_Gh[h,t] == psi[h]*model.Cau_turb[h,t]
model.rest_Gh=pye.Constraint(H,T, rule=regla_gen_hidro)
#Calculo de la potencia de generacion hidraulica
#Definimos Balance hidraulico
def regla_bal_hidro(model,h,t):
    if t == 1:
        return pye.Constraint.Skip
    else:
        return model.Vol[h,t] == model.Vol[h,t-1] + Y[h,t] - model.Cau_turb[h,t] - model.Vert[h,t]
model.rest_b_h = pye.Constraint(H,T, rule = regla_bal_hidro)
#=============================================================================
#==============================Restricciones Termica==========================
#=============================================================================
def regla_RUg(model,g,t):
    if t==1:
        return pye.Constraint.Skip
    else:
        return model.P_Gt[g,t]- model.P_Gt[g,t-1]<= RUt[g]
model.Restriccion_RUg = pye.Constraint(G,T,rule=regla_RUg)

#-----------Rampa de bajada----------------------------------------
def regla_RDg(model,g,t):
    if t==1:
        return pye.Constraint.Skip
    else:
        return model.P_Gt[g,t-1]- model.P_Gt[g,t] <= RDt[g]
model.Restriccion_RDg = pye.Constraint(G,T,rule=regla_RDg)
#=============================================================================
#==============================Restriccion Eolicas============================
#=============================================================================

#------------Rampa de subida---------------------------------------
def regla_RUe(model,e,t):
    if t==1:
        return pye.Constraint.Skip
    else:
        return model.P_Ge[e,t]- model.P_Ge[e,t-1]<= RUe[e]
model.Restriccion_RUe = pye.Constraint(E,T,rule=regla_RUe)

#-----------Rampa de bajada----------------------------------------
def regla_RDe(model,e,t):
    if t==1:
        return pye.Constraint.Skip
    else:
        return model.P_Ge[e,t-1]- model.P_Ge[e,t] <= RDe[e]
model.Restriccion_RDe = pye.Constraint(E,T,rule=regla_RDe)

#Restriccion par la generacion eolica
def regla_Wind(model,t):
    return model.Pw[t]+model.Pwc[t] <= Wind[t]
model.Restriccion_Wind = pye.Constraint(T, rule=regla_Wind)
#------------------------------------------------------------

#=============================================================================
#==============================   Uso de Solver  =============================
#=============================================================================
opt = pye.SolverFactory("glpk")
opt.solve(model,TimeoutError)
results = opt.solve(model, tee=True)
#print("El estado del Programa es:")
termination = results.solver.termination_condition
#print("\nEl programa finalizó debido a que encontró un resultado: ",termination)
#print("\nEstado del programa: ", results.solver.status)
#=============================================================================
#=============================================================================

Gh = {}
Gt = {}
U = {}
for t in T:
    for h in H:
        Gh[h,t] = pye.value(model.P_Gh[h,t]) 
    for g in G:
        Gt[g,t] = pye.value(model.P_Gt[g,t])

xx = model.obj_t

#print("costo de produccion: ",pye.value(model.obj_t),"[$/Dia]")
 
Gsum = {}
Esum = {}
Hsum = {}
Tot={}
if anho ==2021:
    for t in T:
    
        Gsum[t] = pye.value(model.P_Gt["G1",t])+pye.value(model.P_Gt["G2",t])+pye.value(model.P_Gt["G3",t]) \
            + pye.value(model.P_Gt["G4",t])
        
        Esum[t] = pye.value(model.P_Ge["E1",t])+pye.value(model.P_Ge["E2",t])+pye.value(model.P_Ge["E3",t])
    
        Hsum[t] = pye.value(model.P_Gh["H1",t])+pye.value(model.P_Gh["H2",t])+pye.value(model.P_Gh["H3",t])\
            +pye.value(model.P_Gh["H4",t])
    
        Tot[t]=Gsum[t]+Esum[t]+Hsum[t]

elif anho == 2050:
    for t in T:
        Gsum[t] = pye.value(model.P_Gt["G1",t])+pye.value(model.P_Gt["G2",t])
        
        Esum[t] = pye.value(model.P_Ge["E1",t])+pye.value(model.P_Ge["E2",t])+pye.value(model.P_Ge["E3",t])\
                 +pye.value(model.P_Ge["E4",t])+pye.value(model.P_Ge["E5",t])+pye.value(model.P_Ge["E6",t])\
                 +pye.value(model.P_Ge["E7",t])+pye.value(model.P_Ge["E8",t])+pye.value(model.P_Ge["E9",t])
    
        Hsum[t] = pye.value(model.P_Gh["H1",t])+pye.value(model.P_Gh["H2",t])
    
        Tot[t]=Gsum[t]+Esum[t]+Hsum[t]
else:
    print("Año incorrecto")

import matplotlib.pyplot as plt
plt.figure('Generadoras')
plt.plot(Dem.keys(),Dem.values(),label='Demanda') #Linea Azul
plt.plot(Gsum.keys(),Gsum.values(),label='Termicas')
plt.plot(Esum.keys(),Esum.values(),label='Eolicas')
plt.plot(Hsum.keys(),Hsum.values(),label='Hidraulicas')
# plt.plot(Tot.keys(),Tot.values(),label='Tot')
plt.legend()
plt.xlabel("Periodo")
plt.ylabel("Potencia")
plt.title("Demanda en proporcion 1/"+str(prop)+"-"+str(anho))
plt.grid()

# plt.figure('Eolicas')
# #plt.plot(Dem.keys(),Dem.values(),label='Demanda') #Linea Azul

# #plt.plot(Esum.keys(),Esum.values(),label='Eolicas')
# plt.plot(Wind.keys(),Wind.values(),label='Viento')
# # plt.plot(Tot.keys(),Tot.values(),label='Tot')
# plt.legend()
# plt.xlabel("Periodo")
# plt.ylabel("Velocidad")
# plt.grid()


#

import numpy as np

#=====================================================================================
def Limita(X, D, VarMin, VarMax):
    # Keep the coyotes in the search space (optimization problem constraint)
    for abc in range(D):
        X[abc] = max([min([X[abc], VarMax[abc]]), VarMin[abc]])

    return X

def FO(t):
    y = pye.value(sum(model.P_Gt[g,t]*bg[g]+cg[g] for g in G for t in T)+\
                  sum(model.P_Ge[e,t]*be[e]+ce[e]+VWC*model.Pwc[t] for e in E for t in T))

    return y

# from Despacho_HiTerEo import xx


if __name__=="__main__":

    import time
    # Objective function definition
    fobj = FO               # Function
    d = 10                  # Problem dimension
    lu = np.zeros((2, d))   # Boundaires
    lu[0, :] = -10          # Lower boundaires
    lu[1, :] = 10           # Upper boundaries

    # COA parameters
    n_packs = 20            # Number of Packs
    n_coy = 5               # Number of coyotes
    # nfevalmax = 20000       # Stopping criteria: maximum number of function evaluations
    nfevalmax = 20          # Stopping criteria: maximum number of function evaluations

    # Experimanetal variables
    n_exper = 3             # Number of experiments
    t = time.time()         # Time counter (and initial value)
    y = np.zeros(n_exper)   # Experiments costs (for stats.)
    lr=[]
    for i in range(n_exper):
        # Apply the COA to the problem with the defined parameters
        gbest, par = scoa.COA(fobj, lu, nfevalmax, n_packs, n_coy)
        # Keep the global best
        y[i] = gbest
        # Show the result (objective cost and time)
        # print("Experiment ", i+1, ", Best: ", gbest, ", time (s): ", time.time()-t)
        lr.append([i+1,gbest,time.time()-t])
        t = time.time()
    
    dfr = pd.DataFrame(lr,columns =['Num Exp',"Mejor Valor", "Tiempo[s]"])
    #print(dfr)
    # Show the statistics
    # print("Statistics (min., avg., median, max., std.)")
    # print([np.min(y), np.mean(y), np.median(y), np.max(y), np.std(y)])
    
    c1 = ["Minimo","Promedio","Media","Maximo","Desviacion"]
    c2 = [np.min(y), np.mean(y), np.median(y), np.max(y), np.std(y)]
    dfrs = pd.DataFrame(list(zip(c1,c2)), index =['1', '2', '3', '4', '5'],columns =['Datos Estaditicos',"Valor"])
    #print(dfrs)