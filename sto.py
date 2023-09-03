
import numpy as np


def Limita(X, D, VarMin, VarMax):
    # Keep the coyotes in the search space (optimization problem constraint)
#    for abc in range(D):
    for abc in range(D):
       X[abc] = max([min([X[abc], VarMax[abc]]), VarMin[abc]])
    
    return X



def STO(FOBJ, lu, MFEs ,N):

    # Siberian Tiger Optimizacion (STO) for Global Optimization.
    # A nature-inspired metaheuristic proposed by Pavel Trojovský (2022).
    #
    #Trojovsky, P., Dehghani, M., & Hanus, P. (2022). Siberian Tiger Optimization:
    #A new bio-inspired metaheuristic algorithm for solving engineering optimization 
    #problems. IEEE Access, 1. https://doi.org/10.1109/access.2022.3229964
    #
    # Universidad Nacional de Ingeniería (UNI),Rímac, Lima, Perú.
    # sebastian.bedoya.r@uni.pe
    # ------------------------------------------------------------------------
    
    
    # Dimensión de la variables de optimización.
    m = lu.shape[1]
    VarMin = lu[0]
    VarMax = lu[1]
    
    
    # Algorithm parameters
    if N < 3:
        raise Exception("At least 3 coyotes per pack must be used")

    #Número de iteraciones
    
    T=(MFEs-N)/4/N
    
    # Inicialización de posiciones de tigres siberianos (Eq. 1) y (Eq. 2)
    costs = np.zeros((N, 1))
    X = np.tile(VarMin, [N, 1]) +\
              np.random.rand(N, m) * (np.tile(VarMax, [N, 1]) - \
              np.tile(VarMin, [N, 1]))

    # Evalucion de cada tigres siberiano en la FO (Eq. 3)
    
    X_best=[]
    
    for n in range(N):
        costs[n, 0] = FOBJ(X[n, :])
        X_best=X[n,:]
        
        if n!= 0:
            if costs[n,0] <= costs[n-1,0]:
                X_best=X[n,:]
        
    nfeval = N

    
    
    for t in range(T):
        for n in range(N):
            PP=
    