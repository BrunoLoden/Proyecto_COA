
#==========================================================================
#=================Datos de la Unidad termica 2021==========================
#==========================================================================
G = ['G1', 'G2', 'G3', 'G4','G5']
#Coeficientes de funcion de osto
bg = {'G1':18.6, 'G2':12.5, 'G3':14.8, 'G4':19.9, 'G5':19.9, 'G6':19.9}
cg = {'G1':900,  'G2':800,  'G3':700,  'G4':470,  'G5':470,  'G6':470}
    #
Pt_max = {'G1':332.4, 'G2':140, 'G3':100, 'G4': 100,'G5':100}
Pt_min = {'G1':0, 'G2':0, 'G3':0,  'G4':0, 'G5':0}
Qt_max = {'G1':10, 'G2':50, 'G3':40, 'G4':24,'G5':24}
Qt_min = {'G1':-16.9, 'G2':-40, 'G3':0,  'G4':-6, 'G5':-6}
Pg={'G1':232.4, 'G2':40, 'G3':0,  'G4':0, 'G5':0}
Vg={'G1':1.06, 'G2':1.045, 'G3':1.01,  'G4':1.07, 'G5':1.09}



#=============================================================================
#==============================LINEAS=========================================
#=============================================================================
#nodos 
N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

T =   [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
     14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24]

Sb = 100
slack = 13
VOLL = 800


#datos de generación térmica

BusG =   {"G1":1,   "G2":2,  "G3":3, "G4":6, "G5":8}


#=============================================================================
#=============================================================================
#=============================================================================

#datos de las lineas

L = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

BusL = { 1:( 1,  2),  2:( 1,  5),  3:( 2,  3),  4:( 2,  4),  5:( 2,  5),
         6:( 3,  4),  7:( 4, 5),  8:( 4,  7),  9:( 4, 9), 10:( 5, 6),
        11:( 6,  11), 12:( 6,  12), 13:( 6, 13), 14:( 7, 8), 15:( 7, 9),
        16:(9, 10), 17:(9, 14), 18:(10, 11), 19:(12, 13), 20:(13, 14)}
         
r = { 1:0.01938,  2:0.05403,   3:0.04699,  4:0.05811,  5:0.05695,   6:0.06701,   7:0.01335,
      8:0,  9:0, 10:0,  11:0.09498, 12:0.12291, 13:0.06615, 14:0,
     15:0, 16:0.03181, 17:0.12711,  18:0.08205, 19:0.22092, 20:0.1709}

x = { 1:0.05917,  2:0.22304,  3:0.19797,  4:0.17632,  5:0.17388,   6:0.17103,   7:0.04211,
      8:0.20912,  9:0.55618, 10:0.25202,  11:0.1989, 12:0.25581, 13:0.13027, 14:0.17615,
     15:0.11001, 16:0.0845, 17:0.27038,  18:0.19207, 19:0.19988, 20:0.34802}

b = { 1:0.0528,  2:0.0492,  3:0.0438,  4:0.034,  5:0.0346,   6:0.0128,  7:0,  8:0, 9:0, 
	 10:0,  11:0, 12:0, 13:0, 14:0, 15:0,16:0, 17:0, 18:0, 
	 19:0, 20:0}

Llim = { 1:200,  2:220,   3:220,  4:220,  5:220,  6:220,  7:600,  8:220,   9:220,  10:300, 11:300, 12:220, 
	    13:220, 14:600,  15:600, 16:600, 17:600, 18:625, 19:625, 20:625,  21:625,  22:625, 23:625, 
	    24:625, 25:1250, 26:625, 27:625, 28:625, 29:625, 30:625, 31:1250, 32:1250, 33:1250, 34:625}

Bshunt = {1:0,  2:0,   3:0,   4:0,   5:0,   6:0,   7:0,   8:0, 
    	   9:0.19, 10:0, 11:0, 12:0, 13:0, 14:0}

for l in L:
  Llim[l]=Llim[l]/100


#datos de la demanda
D    = ["d1",     "d2",    "d3",     "d4",    "d5",    "d6",     "d7",     "d8",
        "d9",     "d10",   "d11"]

BusD = {"d1":2,   "d2":3,  "d3":4,   "d4":5,  "d5":6,  "d6":9,   "d7":10,   "d8":11,
        "d9":12,   "d10":13,"d11":14}

Pdmax= {"d1":21.7, "d2":94.2, "d3":47.8, "d4":7.6, "d5":11.2, "d6":29.5, "d7":9, "d8":3.5,
        "d9":6.1, "d10":13.5,"d11":14.9}

Qdmax= {"d1":12.7, "d2":19, "d3":-3.9, "d4":1.6, "d5":7.5, "d6":16.6, "d7":5.8, "d8":1.8,
        "d9":1.6, "d10":5.8,"d11":5}


LoadB = { "d1":3685, "d2":3644,   "d3":3613,   "d4":3600,  "d5":3589,    "d6":3598,  "d7":3627,  "d8":3652, 
          "d9":3706, "d10":3787, "d11":3839,  "d12":3853, "d13":3871, "d14":3834, "d15":3817, "d16":3819,
         "d17":3874, "d18":1000, "d19":3984, "d20":3936, "d21":3888, "d22":3809, "d23":3746, "d24":3733}

#Taps

Taps=["Tr1","Tr2","Tr3"]

LineTrafo={ 1:"L1",  2:"L1",  3:"L1",  4:"L1",  5:"L1",
         6:"L1",  7:"L1",  8:"Tr1",  9:"Tr2", 10:"Tr3",
        11:"L1", 12:"L1", 13:"L1", 14:"L1", 15:"L1",
        16:"L1", 17:"L1", 18:"L1", 19:"L1", 20:"L1"}




cond={}
sub={}
for l in L:
    cond[l]=r[l]/(r[l]**2+x[l]**2)
    sub[l]=-x[l]/(r[l]**2+x[l]**2)
LB = {}
LR = {}

for d in D:
        LB[d] = Pdmax[d]
        LR[d]=Qdmax[d]


import pyomo.environ as pye

model = pye.ConcreteModel()


# Sets de Número de Barras y Tiempo
N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 
     14]
T =   [12]


#--------------Sets-----------

model.g=pye.Set(initialize=G)
model.l=pye.Set(initialize=L)
model.n=pye.Set(initialize=N)





#=============================================================================
#=============                  Variables                =====================
#=============================================================================


#-----------Potencia Activa y Reactiva de Generación térmica----------------------------
# Definimos limites de esta variable
def bounds_Gt(model,g):
    return (Pt_min[g]/100,Pt_max[g]/100)

model.P_Gt = pye.Var(model.g,bounds=bounds_Gt)


#sin redespacho:

for g in model.g:
  if g !="G1":
        model.P_Gt[g].fix(Pg[g]/100)


def bounds_Qgt(model,g):
  return (Qt_min[g]/100, Qt_max[g]/100)

model.Q_Gt = pye.Var(model.g,bounds=bounds_Qgt)


#-----------Flujo por las líneas--------------------------------------
#Definimos límites
def bound_Pij(model,l):#(Funcion de Regla)
    return (-Llim[l],Llim[l])

model.Pijde = pye.Var(model.l,domain=pye.Reals)
model.Pijpara = pye.Var(model.l,domain=pye.Reals)
model.Qijde = pye.Var(model.l,domain=pye.Reals)
model.Qijpara = pye.Var(model.l,domain=pye.Reals)


#-----------Ángulo de barra-------------------------------------------
model.theta = pye.Var(model.n,bounds=(-1.5,1.5),initialize=0)#Radianes

#Slack
model.theta[1].fix(0)

#-----------Tensión-------------------------------------------

model.V = pye.Var(model.n, bounds=(0.9, 1.1),initialize=1)

#Barras PV


for n in model.n:
    for g in model.g:
        if n==BusG[g]:
            model.V[n].fix(Vg[g])



#=============================================================================
#=================   Restricciones de igualdad y desigualdad   ===============
#=============================================================================

#-----Restricción Flujo de potencia AC por las líneas----------------------
def regla_FuljoPde(m,l):
    i,j = BusL[l]
    return  cond[l]*m.V[i]**2-\
            m.V[i]*m.V[j]*cond[l]*pye.cos(m.theta[i]-m.theta[j])\
            -m.V[i]*m.V[j]*sub[l]*pye.sin(m.theta[i]-m.theta[j]) -m.Pijde[l] == 0

model.rest_FlujoPde = pye.Constraint(model.l,rule = regla_FuljoPde)



def regla_FuljoPpara(m,l):
    i,j = BusL[l]
    return  cond[l]*m.V[j]**2\
           -m.V[i]*m.V[j]*cond[l]*pye.cos( m.theta[j] - m.theta[i] )\
           -m.V[i]*m.V[j]*sub[l]*pye.sin( m.theta[j] - m.theta[i] ) - m.Pijpara[l] == 0

model.rest_FlujoPpara = pye.Constraint(model.l,rule = regla_FuljoPpara)



def regla_FuljoQde(m,l):
    i,j = BusL[l]
    return  -(sub[l]+b[l])*m.V[i]**2\
           - m.V[i]*m.V[j]*pye.sin(m.theta[i]-m.theta[j])*cond[l]\
           + m.V[i]*m.V[j]*sub[l]*pye.cos(m.theta[i]-m.theta[j]) - m.Qijde[l] == 0

model.rest_FlujoQde = pye.Constraint(model.l,rule = regla_FuljoQde)

def regla_FuljoQpara(m,l):
    i,j = BusL[l]
    return -(sub[l]+b[l])*m.V[j]**2\
           - m.V[i]*m.V[j]*pye.sin(m.theta[j]-m.theta[i])*cond[l]\
           + m.V[i]*m.V[j]*sub[l]*pye.cos(m.theta[j]-m.theta[i]) - m.Qijpara[l] == 0

model.rest_FlujoQpara = pye.Constraint(model.l ,rule = regla_FuljoQpara)



def Balance_P_energia(m,n):
    return  sum(m.P_Gt[g] for g in model.g if (n == BusG[g]))\
            - sum(round(LB[d]/100,3) for d in D if (n == BusD[d]) )\
            - sum(m.Pijde[l] for l in model.l if (n == BusL[l][0]))\
            - sum(m.Pijpara[l] for l in model.l if (n == BusL[l][1]))== 0

model.Balance = pye.Constraint(model.n, rule = Balance_P_energia)

def Balance_Q_energia(m,n):
    return  sum(m.Q_Gt[g] for g in G if (n == BusG[g]))\
            - sum(m.Qijde[l]    for l in L if (n == BusL[l][0]))\
            - sum(m.Qijpara[l]    for l in L if (n == BusL[l][1]))\
            - sum(round(LR[d]/100,3)   for d in D if (n == BusD[d]))\
            + m.V[n]**2*Bshunt[n]== 0

model.BalanceQ = pye.Constraint(model.n, rule = Balance_Q_energia)

#=============================================================================
#===========================   Función Objetivo   ============================
#=============================================================================
 
model.obj_t = pye.Objective(expr =sum(model.P_Gt[g] for g in G))

model.pprint()
#=============================================================================
#========================== Solución del Modelo  =============================
#=============================================================================
import time
tglpk = time.time()
opt = pye.SolverFactory("ipopt")

results = opt.solve(model, tee=True)
print("\nEl estado del Programa es:")
termination = results.solver.termination_condition
print("El programa finalizó debido a que encontró un resultado: ",termination)
print("Estado del programa: ", results.solver.status)

r_glpk=pye.value(model.obj_t)

print(r_glpk)

import pandas as pd

df=pd.DataFrame(columns=["G.P. Activa","G.P. Reactiva","Demanda Activa", "Demanda Reactiva","Flujo De","Flujo Para","Angulo","Tension"],index=N)

#Para t=1

for n in N:

    df.loc[n,"G.P. Activa"]=sum(pye.value(model.P_Gt[g]) for g in G if n == BusG[g])

    df.loc[n,"G.P. Reactiva"]=sum(pye.value(model.Q_Gt[g]) for g in G if n == BusG[g])

    df.loc[n,"Demanda Activa"]=sum(LB[d] for d in D if n == BusD[d])
    
    df.loc[n,"Demanda Reactiva"]=sum(LR[d] for d in D if n == BusD[d])

    df.loc[n,"Flujo De"] = sum(pye.value(model.Pijde[l]) for l in L if n == BusL[l][0])
    
    df.loc[n,"Flujo Para"] = sum(pye.value(model.Pijpara[l]) for l in L if n == BusL[l][1])
    
    df.loc[n,"Angulo"] = pye.value(model.theta[n])
    
    df.loc[n,"Tension"] = pye.value(model.V[n]) 

df.to_excel("ResultadosBarras.xlsx")


flujos=pd.DataFrame(index=L)

for l in L:
    flujos.loc[l,"Desde"]=BusL[l][0]
    flujos.loc[l,"Hasta"]=BusL[l][1]
    flujos.loc[l,"Activa De"]=pye.value(model.Pijde[l])
    flujos.loc[l,"Activa Para"]=pye.value(model.Pijpara[l])
    flujos.loc[l,"Reactiva De"]=pye.value(model.Qijde[l])
    flujos.loc[l,"Reactiva Para"]=pye.value(model.Qijpara[l])
    
flujos.to_excel("ResultadosLineas.xlsx")