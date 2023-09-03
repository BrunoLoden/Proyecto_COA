from __future__ import division
import random
import math

import pandapower.networks as nw
import pandapower as pp
import numpy as np

import commons

import scipy.stats

from typing import Callable, Union, Dict, Any


import numpy

from solution import solution
import time
import massCalculation
import gConstant
import gField
import move








LU=[1.1,1.1,1.1,1.1,1.1,1.13636,0.806451,0.36764,0]
LL=[0.95,0.95,0.95,0.95,0.95,-1.13636,-0.806451,-0.36764,-20]


net14=nw.case14()

boundsA = np.zeros((9,2))

for i in range(9):
    boundsA[i,0]=LL[i]
    boundsA[i,1]=LU[i]

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


def Limita(X, D, VarMin, VarMax):
    # Keep the coyotes in the search space (optimization problem constraint)
#    for abc in range(D):
    for abc in range(D):
       X[abc] = max([min([X[abc], VarMax[abc]]), VarMin[abc]])
    
    return X




def COA(FOBJ, net, lu, nfevalMAX, n_packs=20, n_coy=5,):

    # Coyote Optimization Algorithm (COA) for Global Optimization.
    # A nature-inspired metaheuristic proposed by Juliano Pierezan and
    # Leandro dos Santos Coelho (2018).
    #
    # Pierezan, J. and Coelho, L. S. "Coyote Optimization Algorithm: A new
    # metaheuristic for global optimization problems", Proceedings of the IEEE
    # Congress on Evolutionary Computation (CEC), Rio de Janeiro, Brazil, July
    # 2018, pages 2633-2640.
    #
    # Federal University of Parana (UFPR), Curitiba, Parana, Brazil.
    # juliano.pierezan@ufpr.br
    # ------------------------------------------------------------------------
    iteracion=[]
    # Optimization problem variables
    D = lu.shape[1]
    VarMin = lu[0]
    VarMax = lu[1]

    # Algorithm parameters
    if n_coy < 3:
        raise Exception("At least 3 coyotes per pack must be used")

    # Probability of leaving a pack
    p_leave = 0.005*(n_coy**2)
    Ps = 1/D

    # Packs initialization (Eq. 2)
    pop_total = n_packs*n_coy
    costs = np.zeros((1, pop_total))
    coyotes = np.tile(VarMin, [pop_total, 1]) +\
              np.random.rand(pop_total, D) * (np.tile(VarMax, [pop_total, 1]) - \
              np.tile(VarMin, [pop_total, 1]))
    ages = np.zeros((1, pop_total))
    packs = np.random.permutation(pop_total).reshape(n_packs, n_coy)
    # Evaluate coyotes adaptation (Eq. 3)
    for c in range(pop_total):
        costs[0, c] = FOBJ(coyotes[c, :],net)
    nfeval = pop_total

    # Output variables
    globalMin = np.min(costs[0, :])
    ibest = np.argmin(costs[0, :])
    globalParams = coyotes[ibest, :]

    # Main loop
    year = 1
    while nfeval < nfevalMAX:  # Stopping criteria
        # Update the years counter
        year += 1

        # Execute the operations inside each pack
        for p in range(n_packs):
            # Get the coyotes that belong to each pack
            coyotes_aux = coyotes[packs[p, :], :]
            costs_aux = costs[0, packs[p, :]]
            ages_aux = ages[0, packs[p, :]]

            # Detect alphas according to the costs (Eq. 5)
            ind = np.argsort(costs_aux)
            costs_aux = costs_aux[ind]
            coyotes_aux = coyotes_aux[ind, :]
            ages_aux = ages_aux[ind]
            c_alpha = coyotes_aux[0, :]

            # Compute the social tendency of the pack (Eq. 6)
            tendency = np.median(coyotes_aux, 0)

            #  Update coyotes' social condition
            new_coyotes = np.zeros((n_coy, D))
            for c in range(n_coy):
                rc1 = c
                while rc1 == c:
                    rc1 = np.random.randint(n_coy)
                rc2 = c
                while rc2 == c or rc2 == rc1:
                    rc2 = np.random.randint(n_coy)

                # Try to update the social condition according
                # to the alpha and the pack tendency(Eq. 12)
                new_coyotes[c, :] = coyotes_aux[c, :] + np.random.rand()*(c_alpha - coyotes_aux[rc1, :]) + \
                                    np.random.rand()*(tendency - coyotes_aux[rc2, :])

                # Keep the coyotes in the search space (optimization problem constraint)
                new_coyotes[c, :] = Limita(new_coyotes[c, :], D, VarMin, VarMax)

                # Evaluate the new social condition (Eq. 13)
                new_cost = FOBJ(new_coyotes[c, :],net)
                
                nfeval += 1

                # Adaptation (Eq. 14)
                if new_cost < costs_aux[c]:
                    costs_aux[c] = new_cost
                    coyotes_aux[c, :] = new_coyotes[c, :]

            # Birth of a new coyote from random parents (Eq. 7 and Alg. 1)
            parents = np.random.permutation(n_coy)[:2]
            prob1 = (1-Ps)/2
            prob2 = prob1
            pdr = np.random.permutation(D)
            p1 = np.zeros((1, D))
            p2 = np.zeros((1, D))
            p1[0, pdr[0]] = 1  # Guarantee 1 charac. per individual
            p2[0, pdr[1]] = 1  # Guarantee 1 charac. per individual
            r = np.random.rand(1, D-2)
            p1[0, pdr[2:]] = r < prob1
            p2[0, pdr[2:]] = r > 1-prob2

            # Eventual noise
            n = np.logical_not(np.logical_or(p1, p2))

            # Generate the pup considering intrinsic and extrinsic influence
            pup = p1*coyotes_aux[parents[0], :] + \
                  p2*coyotes_aux[parents[1], :] + \
                  n*(VarMin + np.random.rand(1, D) * (VarMax - VarMin))

            # Verify if the pup will survive
            pup_cost = FOBJ(pup[0, :],net)
            nfeval += 1
            worst = np.flatnonzero(costs_aux > pup_cost)
            if len(worst) > 0:
                older = np.argsort(ages_aux[worst])
                which = worst[older[::-1]]
                coyotes_aux[which[0], :] = pup
                costs_aux[which[0]] = pup_cost
                ages_aux[which[0]] = 0

            # Update the pack information
            coyotes[packs[p], :] = coyotes_aux
            costs[0, packs[p]] = costs_aux
            ages[0, packs[p]] = ages_aux

        # A coyote can leave a pack and enter in another pack (Eq. 4)
        if n_packs > 1:
            if np.random.rand() < p_leave:
                rp = np.random.permutation(n_packs)[:2]
                rc = [np.random.randint(0, n_coy), np.random.randint(0, n_coy)]
                aux = packs[rp[0], rc[0]]
                packs[rp[0], rc[0]] = packs[rp[1], rc[1]]
                packs[rp[1], rc[1]] = aux

        # Update coyotes ages
        ages += 1

        # Output variables (best alpha coyote among all alphas)
        globalMin = np.min(costs[0, :])
        ibest = np.argmin(costs)
        globalParams = coyotes[ibest, :]
        iteracion.append(globalMin)
    return globalMin, globalParams, iteracion



#--- MAIN ---------------------------------------------------------------------+

class Particle:
    def __init__(self,x0):
        self.position_i=[]          # particle position
        self.velocity_i=[]          # particle velocity
        self.pos_best_i=[]          # best position individual
        self.err_best_i=-1          # best error individual
        self.err_i=-1               # error individual

        for i in range(0,num_dimensions):
            self.velocity_i.append(random.uniform(-1,1))
            self.position_i.append(x0[i])

    # evaluate current fitness
    def evaluate(self,costFunc):
        self.err_i=costFunc(self.position_i)

        # check to see if the current position is an individual best
        if self.err_i<self.err_best_i or self.err_best_i==-1:
            self.pos_best_i=self.position_i
            self.err_best_i=self.err_i
                    
    # update new particle velocity
    def update_velocity(self,pos_best_g):
        w=0.5       # constant inertia weight (how much to weigh the previous velocity)
        c1=1        # cognative constant
        c2=2        # social constant
        
        for i in range(0,num_dimensions):
            r1=random.random()
            r2=random.random()
            
            vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
            vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
            self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

    # update the particle position based off new velocity updates
    def update_position(self,bounds):
        for i in range(0,num_dimensions):
            self.position_i[i]=self.position_i[i]+self.velocity_i[i]
            
            # adjust maximum position if necessary
            if self.position_i[i]>bounds[i][1]:
                self.position_i[i]=bounds[i][1]

            # adjust minimum position if neseccary
            if self.position_i[i]<bounds[i][0]:
                self.position_i[i]=bounds[i][0]
        
class PSO():
    def __init__(self,costFunc,x0,bounds,num_particles,maxiter):
        global num_dimensions
        import time
        t = time.time()
        num_dimensions=len(x0)
        err_best_g=-1                   # best error for group
        pos_best_g=[]                   # best position for group

        # establish the swarm
        swarm=[]
        for i in range(0,num_particles):
            swarm.append(Particle(x0))

        # begin optimization loop
        i=0
        iteracion=[]
        while i<maxiter:
            #print i,err_best_g
            # cycle through particles in swarm and evaluate fitness
            for j in range(0,num_particles):
                swarm[j].evaluate(costFunc)
                
                # determine if current particle is the best (globally)
                if swarm[j].err_i<err_best_g or err_best_g==-1:
                    pos_best_g=list(swarm[j].position_i)
                    err_best_g=float(swarm[j].err_i)
            
            # cycle through swarm and update velocities and position
            for j in range(0,num_particles):
                swarm[j].update_velocity(pos_best_g)
                swarm[j].update_position(bounds)
            i+=1
            iteracion.append(err_best_g)
        # print final results
        #print('FINAL:')
        #print(pos_best_g)
        #print(err_best_g)
        print("Usando PSO : ", err_best_g)
        print("Tiempo empleado usando PSO: ", time.time()-t)
        t = time.time()
        import matplotlib.pyplot as plt
        plt.figure(figsize=(5,5))
        x=list(range(len(iteracion)))
        plt.plot(x,iteracion, label= "PSO")



if __name__ == "__PSO__":
    main()



        
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
    
    #print("GSA is optimizing  \""+objf.__name__+"\"")    
    
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
            #print(['At iteration '+ str(l+1)+ ' the best fitness is '+ str(gBestScore)]);
    
    import matplotlib.pyplot as plt
    x=list(range(len(iteracion)))
    plt.plot(x,iteracion, label= "GSA")
    timerEnd=time.time()  
    s.endTime=time.strftime("%Y-%m-%d-%H-%M-%S")
    s.executionTime=timerEnd-timerStart
    s.convergence=convergence_curve
    s.Algorithm="GSA"
    s.objectivefunc=objf.__name__
    
    print("Usando GSA : ", gBestScore)
    print("Tiempo empleado usando GSA: ", s.executionTime)
    
    return s
         



def get_default_params(dim: int):
    """
        Returns the default parameters of the SHADE Differential Evolution Algorithm.
        :param dim: Size of the problem (or individual).
        :type dim: int
        :return: Dict with the default parameters of the SHADE Differential
        Evolution Algorithm.
        :rtype dict
    """
    return {'max_evals': 10000 * dim, 'memory_size': 100,
            'individual_size': dim, 'population_size': 10 * dim,
            'callback': None, 'seed': None, 'opts': None}


def apply(population_size: int, individual_size: int, bounds: np.ndarray,
          func: Callable[[np.ndarray], float], opts: Any,
          memory_size: int, callback: Callable[[Dict], Any],
          max_evals: int, seed: Union[int, None]) -> [np.ndarray, int]:
    """
    Applies the SHADE differential evolution algorithm.
    :param population_size: Size of the population.
    :type population_size: int
    :param individual_size: Number of gens/features of an individual.
    :type individual_size: int
    :param bounds: Numpy ndarray with individual_size rows and 2 columns.
    First column represents the minimum value for the row feature.
    Second column represent the maximum value for the row feature.
    :type bounds: np.ndarray
    :param func: Evaluation function. The function used must receive one
     parameter.This parameter will be a numpy array representing an individual.
    :type func: Callable[[np.ndarray], float]
    :param opts: Optional parameters for the fitness function.
    :type opts: Any type.
    :param memory_size: Size of the internal memory.
    :type memory_size: int
    :param callback: Optional function that allows read access to the state of all variables once each generation.
    :type callback: Callable[[Dict], Any]
    :param max_evals: Number of evaluations after the algorithm is stopped.
    :type max_evals: int
    :param seed: Random number generation seed. Fix a number to reproduce the
    same results in later experiments.
    :type seed: Union[int, None]
    :return: A pair with the best solution found and its fitness.
    :rtype [np.ndarray, int]
    """
    # 0. Check parameters are valid
    if type(population_size) is not int or population_size <= 0:
        raise ValueError("population_size must be a positive integer.")

    if type(individual_size) is not int or individual_size <= 0:
        raise ValueError("individual_size must be a positive integer.")

    if type(max_evals) is not int or max_evals <= 0:
        raise ValueError("max_evals must be a positive integer.")

    if type(bounds) is not np.ndarray or bounds.shape != (individual_size, 2):
        raise ValueError("bounds must be a NumPy ndarray.\n"
                         "The array must be of individual_size length. "
                         "Each row must have 2 elements.")

    if type(seed) is not int and seed is not None:
        raise ValueError("seed must be an integer or None.")

    np.random.seed(seed)
    random.seed(seed)

    # 1. Initialization
    population = commons.init_population(population_size, individual_size, bounds)
    m_cr = np.ones(memory_size) * 0.5
    m_f = np.ones(memory_size) * 0.5
    archive = []
    k = 0
    fitness = commons.apply_fitness(population, func, opts)

    all_indexes = list(range(memory_size))
    max_iters = max_evals // population_size
    iteracion=[]
    for current_generation in range(max_iters):
        # 2.1 Adaptation
        r = np.random.choice(all_indexes, population_size)
        cr = np.random.normal(m_cr[r], 0.1, population_size)
        cr = np.clip(cr, 0, 1)
        cr[cr == 1] = 0
        f = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=population_size)
        f[f > 1] = 0

        while sum(f <= 0) != 0:
            r = np.random.choice(all_indexes, sum(f <= 0))
            f[f <= 0] = scipy.stats.cauchy.rvs(loc=m_f[r], scale=0.1, size=sum(f <= 0))

        p = np.random.uniform(low=2/population_size, high=0.2, size=population_size)

        # 2.2 Common steps
        mutated = commons.current_to_pbest_mutation(population, fitness, f.reshape(len(f), 1), p, bounds)
        crossed = commons.crossover(population, mutated, cr.reshape(len(f), 1))
        c_fitness = commons.apply_fitness(crossed, func, opts)
        population, indexes = commons.selection(population, crossed,
                                                      fitness, c_fitness, return_indexes=True)

        # 2.3 Adapt for next generation
        archive.extend(population[indexes])

        if len(indexes) > 0:
            if len(archive) > memory_size:
                archive = random.sample(archive, memory_size)
            if max(cr) != 0:
                weights = np.abs(fitness[indexes] - c_fitness[indexes])
                weights /= np.sum(weights)
                m_cr[k] = np.sum(weights * cr[indexes])
            else:
                m_cr[k] = 1

            m_f[k] = np.sum(f[indexes]**2)/np.sum(f[indexes])

            k += 1
            if k == memory_size:
                k = 0

        fitness[indexes] = c_fitness[indexes]
        best1 = np.argmin(fitness)
        iteracion.append(fitness[best1])
        if callback is not None:
            callback(**(locals()))

    best = np.argmin(fitness)
    return population[best], fitness[best],iteracion











#--- RUN ----------------------------------------------------------------------+

initial1= np.tile(LL, [1, 1]) +\
          np.random.rand(1, 1) * (np.tile(LU, [1, 1]) - \
          np.tile(LL, [1, 1])) 

initial0=[]

for i in range(9):
    initial0.append(initial1[0,i])

# initial starting location [x1,x2...]


bounds=[(0.95,1.1),(0.95,1.1),(0.95,1.1),(0.95,1.1),(0.95,1.1),(-1.13636,1.13636),(-0.806451,0.806451),(-0.36764,0.36764),(-20,0)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]




if __name__=="__main__":

     import time
     # Objective function definition
     fobj = F1          # Function
     d = 9                  # Problem dimension
     lu = np.zeros((2, d))   # Boundaires
     lu[0,:] = LL[:]          # Lower boundaires
     lu[1,:] = LU[:]          # Upper boundaries

     # COA parameters
     n_packs =  5           # Number of Packs
     n_coy = 5               # Number of coyotes
     nfevalmax = 200*d     # Stopping criteria: maximum number of function evaluations

     # Experimanetal variables
     n_exper = 1             # Number of experiments
     t1 = time.time()         # Time counter (and initial value)
     x = np.zeros(n_exper)
     y = np.zeros(n_exper)   # Experiments costs (for stats.)
     for i in range(n_exper):
         
        # Apply the COA to the problem with the defined parameters
        gbest, par, iteracion1 = COA(fobj,net14, lu, nfevalmax, n_packs, n_coy)
        
        print("Usando COA : ", gbest)
        print("Tiempo empleado usando COA: ", time.time()-t1)
        
        PSO(F1,initial0,bounds,num_particles=50,maxiter=60)
        GSA(F1, np.array(LL), np.array(LU), 9, 50, 60)
        t2=time.time()
        best,para,iteracion=apply(50, 9, boundsA, F1,opts=None,memory_size=100,callback=None,max_evals=3000,seed=None)
        print("Usando SHADE : ", para)
        print("Tiempo empleado usando SHADE: ", time.time()-t2)       
        # Keep the global best
        
        import matplotlib.pyplot as plt
        
        x=list(range(len(iteracion1)))
    
        plt.plot(x,iteracion1,label="COA")
        x=list(range(len(iteracion)))
        plt.plot(x,iteracion,label="SHADE")
        plt.xlabel('Iteración')
        plt.ylabel('Pérdidas de Transmisión (MW)')
        plt.legend(loc='best')
        
        
        