
from utils import ncon
from utils import SVD
from utils import EigenSolver
torch_svd = SVD.apply
torch_eigh = EigenSolver.apply
import numpy as np
import torch
dtype = torch.float64; device = 'cpu'
torch.autograd.set_detect_anomaly(True)
##### Set bond dimensions and temperature
chiM = 20
#relTemp = 0.8  # temp relative to the crit temp, in [0,inf]
log2L = 10 #7
RGstep = 2*log2L # number of coarse-graining

# Tc = (2 / np.log(1 + np.sqrt(2)))
##### Define partition function(classical Ising)
def construct_uv(Atemp, chiM):
    chiA = Atemp.shape[0]
    chitemp = min(chiA ** 2, chiM)
    utemp, stemp, vtemp = torch_svd(Atemp.reshape(chiA ** 2, chiA ** 2))
    vtemp = vtemp.T
    U1 = utemp[:, :chitemp] @ torch.diag(torch.sqrt(stemp[:chitemp]))
    U1 = U1.reshape(chiA, chiA, chitemp)
    V1 = torch.diag(torch.sqrt(stemp[:chitemp])) @ vtemp[:chitemp, :]
    V1 = (V1.reshape(chitemp,chiA,chiA)).permute(1,2,0)
    return U1, V1

def lnz_trg(betaval, chiM, RGstep):
    # Jtemp = torch.zeros((2, 2, 2), dtype=dtype, device=device)
    # Jtemp[0, 0, 0] = 1
    # Jtemp[1, 1, 1] = 1
    p_beta = torch.exp(betaval)
    m_beta = torch.exp(-betaval)
    # print(p_beta)
    Etemp = [p_beta, m_beta, m_beta, p_beta]
    Etemp = torch.stack(Etemp).view(2, 2)
    # Ainit = ncon([Jtemp, Jtemp, Jtemp, Jtemp, Etemp, Etemp, Etemp, Etemp],
    #             [[-1, 8, 1], [-2, 2, 3], [-3, 4, 5], [-4, 6, 7], [1, 2], [3, 4], [5, 6], [7, 8]])
    w, v = torch_eigh(Etemp)
    # print(w)
    Q_sqrt = v @ torch.diag(torch.sqrt(w)) @ torch.inverse(v)
    delta = torch.zeros([2, 2, 2, 2], dtype=dtype, device=device)
    delta[0, 0, 0, 0] = delta[1, 1, 1, 1] = 1
    # print(torch.norm(Q_sqrt@Q_sqrt-Etemp))
    T = ncon([Q_sqrt,Q_sqrt,Q_sqrt,Q_sqrt, delta],
         [[-1,1], [-2,2], [-3,3], [-4,4], [1,2,3,4]])
    Atemp = T
    # test_a = Atemp.norm()
    # test, = torch.autograd.grad(test_a, betaval, create_graph=True)
    # print('fine')
    lnz = 0.0
    for k in range(RGstep):
        print('sizeA = ', Atemp.shape, ', RGstep = ', k + 1)
        Amax = Atemp.abs().max()
        Atemp = Atemp/Amax
        lnz += 2 ** (-k) * torch.log(Amax)
        # # decompose A site
        U1, V1 = construct_uv(Atemp, chiM)
        # decompose B site
        Atemp = Atemp.permute(3,0,1,2)
        U2, V2 = construct_uv(Atemp, chiM)
        Atemp = ncon([V1,U2,V2,U1],[[1,4,-1],[2,1,-2],[4,3,-4],[3,2,-3]])

    lnz += torch.log(ncon([Atemp], [[1, 2, 1, 2]])) / 2 ** (RGstep)
    return lnz


K = 0.4
beta = torch.tensor(K, dtype=dtype, device=device).requires_grad_()
lnz = lnz_trg(beta, chiM, RGstep)
# dlnz, = torch.autograd.grad(lnz, beta,create_graph=True) #  En = -d lnZ / d beta
# print(dlnz)

betaval = beta.item()
Tval = 1/betaval
FreeEnergy = -Tval*lnz
FreeEnergy = FreeEnergy.item()
##### Compare with exact results (thermodynamic limit)
maglambda = 1 / (np.sinh(2 / Tval) ** 2)
N = 1000000;
x = np.linspace(0, np.pi, N + 1)
y = np.log(np.cosh(2 * betaval) * np.cosh(2 * betaval) + (1 / maglambda) * np.sqrt(
        1 + maglambda ** 2 - 2 * maglambda * np.cos(2 * x)))
FreeExact = -Tval * ((np.log(2) / 2) + 0.25 * sum(y[1:(N + 1)] + y[:N]) / N)
RelFreeErr = abs((FreeEnergy - FreeExact) / FreeExact)
print(FreeEnergy)
print('Exact = ', FreeExact)
print('RelFreeErr = ', RelFreeErr)



# filename = 'ising_trg_square.txt'
# with open(filename, 'w') as f:
#     f.write('# correlation_length v.s m \n')
#     f.write('# beta \t dlnz \t beta^2*dlnz2 \n')

# for K in np.linspace(0.4, 0.5, 11):
#     beta = torch.tensor(K, dtype=dtype, device=device).requires_grad_()
#     lnz = lnz_trg(beta, chiM, RGstep)
#     dlnz, = torch.autograd.grad(lnz, beta, create_graph=True)  # En = -d lnZ / d beta
#     dlnz2, = torch.autograd.grad(dlnz, beta)  # Cv = beta^2 * d^2 lnZ / d beta^2
#     # print(K, lnz.item(), -dlnz.item(), dlnz2.item() * beta.item() ** 2)
#     # f = open(filename, 'a')
#     message = ('K={:.3f}, ' + 'lnz={:.5f}, '+'dlnz={:.5f}, '+'dlnz2={:.5f} ')\
#         .format(K, lnz.item(), -dlnz.item(), dlnz2.item() * beta.item() ** 2)
#     print(message)
#     # f.write(message)
#     # f.write('\n')
#     # f.close()
