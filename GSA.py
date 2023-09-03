# -*- coding: utf-8 -*-
"""
Python code of Gravitational Search Algorithm (GSA)
Reference: Rashedi, Esmat, Hossein Nezamabadi-Pour, and Saeid Saryazdi. "GSA: a gravitational search algorithm." 
           Information sciences 179.13 (2009): 2232-2248.	
Coded by: Mukesh Saraswat (saraswatmukesh@gmail.com), Himanshu Mittal (emailid: himanshu.mittal224@gmail.com) and Raju Pal (emailid: raju3131.pal@gmail.com)
The code template used is similar given at link: https://github.com/7ossam81/EvoloPy and matlab version of GSA at mathworks.

Purpose: Main file of Gravitational Search Algorithm(GSA) 
            for minimizing of the Objective Function

Code compatible:
 -- Python: 2.* or 3.*
"""

import random
import numpy
import math
from solution import solution
import time
import massCalculation
import gConstant
import gField
import move



import pandapower.networks as nw
import pandapower as pp
import numpy as np

LU=[1.1,1.1,1.1,1.1,1.1,1.13636,0.806451,0.36764,0]
LL=[0.95,0.95,0.95,0.95,0.95,-1.13636,-0.806451,-0.36764,-20]


net14=nw.case14()



def F1 (X,net=net14):

    net.ext_grid.loc[0,"vm_pu"]=X[0]
    net.gen.loc[0,"vm_pu"]=X[1]
    net.gen.loc[1,"vm_pu"]=X[2]
    net.gen.loc[2,"vm_pu"]=X[3]
    net.gen.loc[3,"vm_pu"]=X[4]
    net.trafo.loc[0,"tap_pos"]=X[5]
    net.trafo.loc[1,"tap_pos"]=X[6]
    net.trafo.loc[2,"tap_pos"]=X[7]
    net.shunt.loc[0,"q_mvar"]=X[8]
    pp.runpp(net)
    perdidas=net.res_line["pl_mw"].sum()
    return perdidas

def F2 (X,net=net14):

    net.ext_grid.loc[0,"vm_pu"]=X[0]
    net.gen.loc[0,"vm_pu"]=X[1]
    net.gen.loc[1,"vm_pu"]=X[2]
    net.gen.loc[2,"vm_pu"]=X[3]
    net.gen.loc[3,"vm_pu"]=X[4]
    net.trafo.loc[0,"tap_pos"]=X[5]
    net.trafo.loc[1,"tap_pos"]=X[6]
    net.trafo.loc[2,"tap_pos"]=X[7]
    net.shunt.loc[0,"q_mvar"]=X[8]
    pp.runpp(net)
    VD=abs(net.res_bus["vm_pu"]-1).sum()
    return VD








        
def GSA(objf,lb,ub,dim,PopSize,iters):
    # GSA parameters
    ElitistCheck =1
    Rpower = 1 
     
    s=solution()
        
    """ Initializations """
    
    vel=numpy.zeros((PopSize,dim))
    fit = numpy.zeros(PopSize)
    M = numpy.zeros(PopSize)
    gBest=numpy.zeros(dim)
    gBestScore=float("inf")
    
    pos=numpy.random.uniform(0,1,(PopSize,dim)) *(ub-lb)+lb
    
    convergence_curve=numpy.zeros(iters)
    
    print("GSA is optimizing  \""+objf.__name__+"\"")    
    
    timerStart=time.time() 
    s.startTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    iteracion=[]
    for l in range(0,iters):
        for i in range(0,PopSize):
            l1 = [None] * dim
            l1=numpy.clip(pos[i,:], lb, ub)
            pos[i,:]=l1

            #Calculate objective function for each particle
            fitness=[]
            fitness=objf(l1)
            fit[i]=fitness
    
                
            if(gBestScore>fitness):
                gBestScore=fitness
                gBest=l1           
        
        """ Calculating Mass """
        M = massCalculation.massCalculation(fit,PopSize,M)

        """ Calculating Gravitational Constant """        
        G = gConstant.gConstant(l,iters)        
        
        """ Calculating Gfield """        
        acc = gField.gField(PopSize,dim,pos,M,l,iters,G,ElitistCheck,Rpower)
        
        """ Calculating Position """        
        pos, vel = move.move(PopSize,dim,pos,vel,acc)
        
        convergence_curve[l]=gBestScore
        
        if (l%1==0):
            iteracion.append(gBestScore)   
            print(['At iteration '+ str(l+1)+ ' the best fitness is '+ str(gBestScore)]);
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.Algorithm="GSA"
    s.objectivefunc=objf.__name__

    return s
         
    
s=GSA(F1, np.array(LL), np.array(LU), 9, 50, 60)