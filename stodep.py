import numpy as np


def Sphere(t):
    y=t[0]**2+t[1]**2+t[2]**2+t[3]**2+t[4]**2
    return y



FOBJ = Sphere           # Function
m = 5                  # Problem dimension
lu = np.zeros((2, m))   # Boundaires
lu[0, :] = -10          # Lower boundaires
lu[1, :] = 10           # Upper boundaries
# Dimensión de la variables de optimización.
m = lu.shape[1]
VarMin = lu[0]
VarMax = lu[1]

N=5
MFEs=10000*m

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


for n in range(N):
    costs[n, 0] = FOBJ(X[n, :])

X_best=X[np.where(costs == np.amin(costs))[0],:] 

nfeval = N