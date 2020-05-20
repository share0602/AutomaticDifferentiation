import jax.numpy as np
from jax.numpy import linalg as LA
import jax
from jax_ncon import ncon
from jax import grad, jit, vmap
from jax.config import config
config.update("jax_debug_nans", True)
##### Set bond dimensions and temperature
chiM = 4
#relTemp = 0.8  # temp relative to the crit temp, in [0,inf]
log2L = 8 #7 will be nan # number of coarse-graining
Tc = (2 / np.log(1 + np.sqrt(2)))
relTemp = 0.8 
Tval = relTemp*Tc
betaval = 1/Tval
#betaval = 0.4
def construct_uv(Atemp, chiM):
    chiA = Atemp.shape[0]
    chitemp = min(chiA ** 2, chiM)
    utemp, stemp, vtemp = LA.svd(Atemp.reshape(chiA ** 2, chiA ** 2), full_matrices=False)
    U1 = utemp[:, :chitemp] @ np.diag(np.sqrt(stemp[:chitemp]))
    U1 = U1.reshape(chiA, chiA, chitemp)
    V1 = np.diag(np.sqrt(stemp[:chitemp])) @ vtemp[:chitemp, :]
    V1 = np.transpose(V1.reshape(chitemp, chiA, chiA), (1, 2, 0))
    return U1, V1

##### Define partition function(classical Ising)
def lnz_trg(betaval, chiM, log2L):
    RGstep = 2 * log2L - 1
    Jtemp = np.zeros((2, 2, 2))
    Jtemp = jax.ops.index_update(Jtemp,jax.ops.index[0,0,0],1.)
    Jtemp = jax.ops.index_update(Jtemp,jax.ops.index[1,1,1],1.)
    Etemp = np.array([[np.exp(betaval), np.exp(-betaval)], [np.exp(-betaval), np.exp(betaval)]])
    Ainit = ncon([Jtemp, Jtemp, Jtemp, Jtemp, Etemp, Etemp, Etemp, Etemp],
                [[-1, 8, 1], [-2, 2, 3], [-3, 4, 5], [-4, 6, 7], [1, 2], [3, 4], [5, 6], [7, 8]])
    Atemp = Ainit
    lnz = 0.0
    for k in range(RGstep):
        Amax = np.max(np.abs(Atemp))
        Atemp /= Amax
        lnz += 2**(-1-k)*np.log(Amax)
        # decompose A site
        U1,V1 = construct_uv(Atemp, chiM)
        # decompose B site
        Atemp = np.transpose(Atemp,(3,0,1,2))
        U2, V2 = construct_uv(Atemp,chiM)
        Atemp = ncon([V1,U2,V2,U1],[[1,4,-1],[2,1,-2],[4,3,-4],[3,2,-3]])
        # extract result
        print('sizeA = ',Atemp.shape, ', RGstep = ', k+1)
    lnz += np.log(ncon([Atemp], [[1, 2, 1, 2]]))/2**(RGstep+1)
    return lnz

print('Calculate lnz')
lnz = lnz_trg(betaval, chiM, log2L)
Tval = 1/betaval
FreeEnergy = -Tval*lnz
##### Compare with exact results (thermodynamic limit)
maglambda = 1 / (np.sinh(2 / Tval) ** 2)
N = 1000000;
x = np.linspace(0, np.pi, N + 1)
y = np.log(np.cosh(2 * betaval) * np.cosh(2 * betaval) + (1 / maglambda) * np.sqrt(
        1 + maglambda ** 2 - 2 * maglambda * np.cos(2 * x)))
FreeExact = -Tval * ((np.log(2) / 2) + 0.25 * sum(y[1:(N + 1)] + y[:N]) / N)
RelFreeErr = abs((FreeEnergy - FreeExact) / FreeExact)
print('FreeEnergy= %.5f'%FreeEnergy)
print('Exact = ', FreeExact)
print('RelFreeErr = ', RelFreeErr)

print('Calculate dlnz')
dlnz = grad(lnz_trg, argnums=0)(betaval, chiM, log2L)
#dlnz = grad(lnz_trg)(betaval)
print('dlnz: %.5f'%dlnz)
#from jax import value_and_grad
#lnz, dlnz = value_and_grad(lnz_trg, argnums=0)(betaval, chiM, log2L)
#dlnZ2 = grad(grad(lnz_trg))(betaval)
#print(dlnZ2)
#print(lnz, dlnz)
#for betaval in np.linspace(0.4, 0.5, 11):
    #lnz,dlnz = value_and_grad(lnz_trg, argnums=0)(betaval, chiM, log2L)
    #print(lnz,dlnz)
    #dlnz2 = grad(grad(lnz_trg))(betaval, chiM, log2L)
    #print(dlnz2)
