

import commons
import numpy as np
import scipy.stats
import random
from typing import Callable, Union, Dict, Any

import pandapower.networks as nw
import pandapower as pp
import numpy as np




LU=[1.1,1.1,1.1,1.1,1.1,1.13636,0.806451,0.36764,0]
LL=[0.95,0.95,0.95,0.95,0.95,-1.13636,-0.806451,-0.36764,-20]


net14=nw.case14()

bounds = np.zeros((9,2))

for i in range(9):
    bounds[i,0]=LL[i]
    bounds[i,1]=LU[i]

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


best,para,iteracion=apply(50, 9, bounds, F1,opts=None,memory_size=100,callback=None,max_evals=2000,seed=None)


import matplotlib.pyplot as plt

x=list(range(len(iteracion)))

plt.plot(x,iteracion,label="SHADE")

plt.xlabel('Iteración')
plt.ylabel('Pérdias de Transmisión (MW)')
plt.legend(loc='best')
