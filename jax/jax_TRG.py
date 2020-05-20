import jax.numpy as np
from numpy import linalg as LA
import jax
from jax_ncon import ncon

##### Set bond dimensions and temperature
chiM = 10
relTemp = 0.8  # temp relative to the crit temp, in [0,inf]
log2L = 5  # number of coarse-graining
RGstep = 2*log2L

##### Define partition function(classical Ising)
Tc = (2 / np.log(1 + np.sqrt(2)))
Tval = relTemp * Tc
betaval = 1 / Tval
Jtemp = np.zeros((2, 2, 2))
Jtemp = jax.ops.index_update(Jtemp,jax.ops.index[0,0,0],1.)
Jtemp = jax.ops.index_update(Jtemp,jax.ops.index[1,1,1],1.)
#Jtemp[0, 0, 0] = 1
#Jtemp[1, 1, 1] = 1
Etemp = np.array([[np.exp(betaval), np.exp(-betaval)], [np.exp(-betaval), np.exp(betaval)]])
Ainit = ncon([Jtemp, Jtemp, Jtemp, Jtemp, Etemp, Etemp, Etemp, Etemp],
             [[-1, 8, 1], [-2, 2, 3], [-3, 4, 5], [-4, 6, 7], [1, 2], [3, 4], [5, 6], [7, 8]])
Atemp = Ainit
Anorm = np.ones(RGstep + 1)
Anorm = jax.ops.index_update(Anorm,jax.ops.index[0],LA.norm(Atemp.flatten()))
#Anorm[0] = LA.norm(Atemp.flatten())
A = [0 for x in range(RGstep + 1)]
Atemp = Atemp/Anorm[0]
#jax.ops.index_update(A,jax.ops.index[0],Atemp)
A[0] = Atemp
for k in range(RGstep):
    chiA = A[k].shape[0]
    # decompose A site
    chitemp = min(chiA**2, chiM)
    utemp,stemp,vtemp = LA.svd(Atemp.reshape(chiA**2,chiA**2))
    U1 = utemp[:, :chitemp]@np.diag(np.sqrt(stemp[:chitemp]))
    U1 = U1.reshape(chiA, chiA, chitemp)
    V1 = np.diag(np.sqrt(stemp[:chitemp]))@vtemp[:chitemp,:]
    V1 = np.transpose(V1.reshape(chitemp,chiA,chiA),(1,2,0))
    # decompose B site
    test = np.transpose(Atemp,(3,0,1,2))
    utemp,stemp,vtemp = LA.svd(np.reshape(test,(chiA**2,chiA**2)))
    U2 = utemp[:,:chitemp]@np.diag(np.sqrt(stemp[:chitemp]))
    U2 = np.reshape(U2,(chiA,chiA,chitemp))
    V2 = np.diag(np.sqrt(stemp[:chitemp]))@vtemp[:chitemp,:]
    V2 = np.transpose(np.reshape(V2,(chitemp,chiA,chiA)),(1,2,0))
    Atemp = ncon([V1,U2,V2,U1],[[1,4,-1],[2,1,-2],[4,3,-4],[3,2,-3]])
    # extract result
    #Anorm[k+1] = LA.norm(Atemp.flatten())
    Anorm = jax.ops.index_update(Anorm,jax.ops.index[k+1], LA.norm(Atemp.flatten()))
    Atemp = Atemp/Anorm[k+1]
    A[k+1] = Atemp
    #print(Anorm)
    print('sizeA = ',Atemp.shape, ', RGstep = ', k+1)
print(Anorm)
# %%%%%% Evaluate Free Energy
# % recall k = 2log2L = RGstep
import numpy as onp
#NumSpins = 2 ** (onp.array(range(1,50)))
NumSpins = 2**(np.array([x for x in range(1,50)]))
FreeEnergy = -Tval * (sum((2 ** np.linspace(k+1,0,k+2) )* np.log(Anorm[:(k + 2)])) +
                                     np.log(ncon([A[k+1]], [[1, 2, 1, 2]]))) / NumSpins[k+1]
print(FreeEnergy)
##### Compare with exact results (thermodynamic limit)
maglambda = 1 / (np.sinh(2 / Tval) ** 2)
N = 1000000;
x = np.linspace(0, np.pi, N + 1)
y = np.log(np.cosh(2 * betaval) * np.cosh(2 * betaval) + (1 / maglambda) * np.sqrt(
        1 + maglambda ** 2 - 2 * maglambda * np.cos(2 * x)))
FreeExact = -Tval * ((np.log(2) / 2) + 0.25 * sum(y[1:(N + 1)] + y[:N]) / N)
RelFreeErr = abs((FreeEnergy - FreeExact) / FreeExact)
print('RelFreeErr = ', RelFreeErr)
#print(Anorm)
