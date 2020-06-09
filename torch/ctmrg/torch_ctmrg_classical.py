import sys
sys.path.insert(0, '../')
from utils import ncon
from utils import QR
from utils import SVD
from utils import EigenSolver
torch_qr = QR.apply
torch_svd = SVD.apply
torch_eigh = EigenSolver.apply
import numpy as np
import torch
dtype = torch.float64; device = 'cpu'
torch.autograd.set_detect_anomaly(True)

def list_rotation(cns):
    """Returns new tuple shifted to left by one place."""
    cns = [cns[1], cns[2], cns[3], cns[0]]
    return cns
def weight_rotate(weight):
    """Returns weight rotated anti-clockwise."""
    weight = weight.permute(1, 2, 3, 0)
    return weight

def extend_tensors(c1, t1, t4, w, c4, t3):
    ## c1t1
    chi0, chi1, chi2 = (t1.shape[0], c1.shape[1], t1.shape[1])
    c1t1 = ncon([c1,t1],
                [[1,-2], [-1,-3,1]])
    c1t1 = c1t1.reshape([chi0, chi1*chi2])
    ## t4w
    chi0, chi1,chi2,chi3,chi4 = (t4.shape[0], w.shape[0],w.shape[0], t4.shape[2], w.shape[0])
    t4w = ncon([t4, w],
                [[-1,1,-4], [-2,-3,-5,1]])
    t4w = t4w.reshape([chi0*chi1, chi2, chi3*chi4])
    ## c4t3
    chi0, chi1, chi2 = (c4.shape[0], t3.shape[1], t3.shape[2])
    c4t3 = ncon([c4, t3],
                [[-1, 1], [1, -2, -3]])
    c4t3 = c4t3.reshape([chi0*chi1, chi2])
    return c1t1, t4w, c4t3

def create_projectors_orus(c1t1, c4t3, dim_cut):
    dim_new = min(c1t1.shape[0], dim_cut)
    m1 = c1t1.T @ c1t1; m2 = c4t3 @ c4t3.T;
    w, u = torch_eigh(m1 + m2)
    u = torch.flip(u, [1])
    u = u[:, :dim_cut]
    u_up = u;
    u_down = u
    return u_up, u_down

def corner_extend_corboz(c1, t1, t4, w):
    c1_label = [2,1]; t1_label = [-1,3,2];
    t4_label = [1,4,-3]; w_label = [3,-2,-4,4]
    c1_new = ncon([c1,t1,t4,w], [c1_label, t1_label, t4_label, w_label])
    dim1 = c1_new.shape[0]*c1_new.shape[1]
    dim2 = c1_new.shape[2]*c1_new.shape[3]
    c1_new = c1_new.reshape([dim1, dim2])
    return c1_new

def create_projectors_corboz(c1, c2, c3, c4, dim_cut):
    upper_half = ncon([c1, c2],
                      [[1,-2] , [-1,1]])
    lower_half = ncon([c4,c3],
                      [[-2,1],[1,-1]])

    _, r_up = torch_qr(upper_half)
    _, r_down = torch_qr(lower_half)
    rr = r_up@ r_down.T.contiguous()
    u, s, v = torch_svd(rr)
    vT = v.T
    dim_new = min(s.shape[0], dim_cut)
    s = s[:dim_new]
    s_inv2 = torch.diag(1 / s ** 0.5)
    u_tmp = u[:, :dim_new]@ s_inv2
    v_tmp = s_inv2 @ vT[:dim_new, :]
    p_up = r_down.T @ v_tmp.T
    p_down = r_up.T @ u_tmp
    return p_up, p_down

def coarse_grain(c1t1, t4w, c4t3, u_up, u_down):
    c1 = c1t1 @ u_up
    t4 = ncon([u_down, t4w, u_up],
              [[1, -1], [1, -2, 2], [2, -3]])
    c4 = u_down.T.contiguous() @ c4t3
    return c1, t4, c4
def normalize(x):
    return x/x.norm()
def free_energy_ctmrg(lattice, beta, chiM, CTMRGstep, thres = 1e-7):
    p_beta = torch.exp(beta)
    m_beta = torch.exp(-beta)
    # print(p_beta)
    Q = [p_beta, m_beta, m_beta, p_beta]
    Q = torch.stack(Q).view(2, 2)
    w, v = torch_eigh(Q)
    # print(w)
    Q_sqrt = v @ torch.diag(torch.sqrt(w)) @ torch.inverse(v)
    if lattice == 'square':
        delta = torch.zeros([2, 2, 2, 2], dtype=dtype, device=device)
        delta[0, 0, 0, 0] = delta[1, 1, 1, 1] = 1
        # print(torch.norm(Q_sqrt @ Q_sqrt - Q))
        weight = ncon([Q_sqrt, Q_sqrt, Q_sqrt, Q_sqrt, delta],
                 [[-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]])
        s_mag = torch.zeros([2, 2, 2, 2], dtype=dtype, device=device)
        s_mag[0, 0, 0, 0] = 1; s_mag[1, 1, 1, 1] = -1
        weight_mag = ncon([Q_sqrt, Q_sqrt, Q_sqrt, Q_sqrt, s_mag],
                 [[-1, 1], [-2, 2], [-3, 3], [-4, 4], [1, 2, 3, 4]])
    elif lattice == 'honeycomb':
        delta = torch.zeros([2, 2, 2], dtype=dtype, device=device)
        delta[0,0,0] = delta[1,1,1] = 1
        ten_a = ncon([Q_sqrt, Q_sqrt, Q_sqrt, delta],
                      [[-1, 1], [-2, 2], [-3, 3], [1, 2, 3]])
        weight = ncon([ten_a, ten_a],
                      [[1,-3,-4], [1,-1,-2]])
        s_mag = torch.zeros([2, 2, 2], dtype=dtype, device=device)
        s_mag[0, 0, 0] = 1; s_mag[1, 1, 1] = -1
        a_mag = ncon([Q_sqrt, Q_sqrt, Q_sqrt, s_mag],
                          [[-1, 1], [-2, 2], [-3, 3], [1, 2, 3]])
        weight_mag = ncon([a_mag, a_mag],
                      [[1,-3,-4], [1,-1,-2]])
    cns = [torch.rand([2, 2], dtype=dtype, device=device)] * 4
    tms = [torch.rand([2, 2, 2], dtype=dtype, device=device)] * 4
    lnz_val = 0
    lnz_mem_val = -1
    steps = 0
    while abs(lnz_val - lnz_mem_val) > thres and steps < CTMRGstep:
    # while steps < CTMRGstep:
        # print('c1.shape = ', tuple(c1.shape), ', CTMRGstep = ', steps + 1)
        for i in range(4):  # four direction
            ######################## Extend tensors
            c1t1, t4w, c4t3 = extend_tensors(cns[0], tms[0], tms[3], weight, cns[3], tms[2])
            # print(c1t1.shape)
            ######################## Create projector
            # Orus scheme
            u_up, u_down = create_projectors_orus(c1t1, c4t3, chiM)
            ## Corboz scheme
            # corner_extended = []
            # for _ in range(4):
            #     corner_extended.append(corner_extend_corboz(c1, t1, t4, weight))
            #     c1, c2, c3, c4 = tuple_rotation(c1, c2, c3, c4)
            #     t1, t2, t3, t4 = tuple_rotation(t1, t2, t3, t4)
            #     weight = weight_rotate(weight)
            # u_up, u_down = create_projectors_corboz(*corner_extended, chiM)
            ###################### Coarse grain tensors
            cns[0], tms[3], cns[3] = coarse_grain(c1t1, t4w, c4t3, u_up, u_down)

            ###################### Rotate tensors
            cns, tms = map(list_rotation, (cns, tms))
            weight = weight_rotate(weight)
        cns = list(map(normalize, cns))
        tms = list(map(normalize, tms));
        c1, c2, c3, c4 = cns
        t1, t2, t3, t4 = tms
        # print(t4.shape)
        expect1 = ncon([c1, t1, c2, t4, weight, t2, c4, t3, c3],
                       [[1, 3], [2, 4, 1], [5, 2], [3, 6, 8], [4, 7, 9, 6], [10, 7, 5], [8, 11], [11, 9, 12], [12, 10]])
        expect2 = ncon([c1, c2, c3, c4],
                       [[1, 2], [3, 1], [4, 3], [2, 4]])
        expect3 = ncon([c1, t1, c2, c3, t3, c4],
                       [[1, 3], [2, 4, 1], [5, 2], [7, 5], [6, 4, 7], [3, 6]])
        expect4 = ncon([c1, c2, t4, t2, c4, c3],
                       [[1, 2], [3, 1], [2, 4, 5], [6, 4, 3], [5, 7], [7, 6]])
        expect_mag = ncon([c1, t1, c2, t4, weight_mag, t2, c4, t3, c3],
                       [[1, 3], [2, 4, 1], [5, 2], [3, 6, 8], [4, 7, 9, 6], [10, 7, 5], [8, 11], [11, 9, 12], [12, 10]])
        lnz = torch.log(expect1 * expect2 / (expect3 * expect4))
        steps += 1
        lnz_mem_val = lnz_val
        lnz_val = lnz
        mag = expect_mag/expect1
        # print('lnz = ', lnz.item())
        # print('(lnz-lnz_old) = ', abs(lnz_val - lnz_mem_val).item()/lnz_val.item())
    if steps < CTMRG_step:
        print('converge!')
    if lattice == 'honeycomb':
        lnz = lnz/2; mag = mag/2
    print('CTMRGstep = ', steps+1)
    return lnz, mag

##### Set bond dimensions and temperature
lattice = 'square'
CTMRG_step = 500
chiM = 24
print('lattice = ', lattice, 'chiM = ', chiM)
#relTemp = 0.8  # temp relative to the crit temp, in [0,inf]

K = 0.44
beta = torch.tensor(K, dtype=dtype, device=device).requires_grad_()
# beta = torch.tensor(K, dtype=dtype, device=device)
Tval = 1/beta.item()
print('Tval = ', Tval)
lnz, mag = free_energy_ctmrg(lattice, beta,chiM, CTMRG_step)
# dlnz, = torch.autograd.grad(lnz, beta,create_graph=True) #  En = -d lnZ / d beta
# print(dlnz)

print('mag = ', mag.item())
FreeEnergy = -(Tval*lnz).item()
print('FreeEnergy = ', FreeEnergy)

##### Compare with exact results (thermodynamic limit)
maglambda = 1 / (np.sinh(2 / Tval) ** 2)
N = 1000000;
x = np.linspace(0, np.pi, N + 1)
y = np.log(np.cosh(2 * beta.item()) * np.cosh(2 * beta.item()) + (1 / maglambda) * np.sqrt(
        1 + maglambda ** 2 - 2 * maglambda * np.cos(2 * x)))
FreeExact = -Tval * ((np.log(2) / 2) + 0.25 * sum(y[1:(N + 1)] + y[:N]) / N)
RelFreeErr = abs((FreeEnergy - FreeExact) / FreeExact)
# print(FreeEnergy)
print('Exact = ', FreeExact)
print('RelFreeErr = ', RelFreeErr)
# dlnz, = torch.autograd.grad(lnz, beta, create_graph=True) #  En = -d lnZ / d beta
# dlnz2, = torch.autograd.grad(dlnz, beta)
# message = ('K={:.3f}, ' + 'lnz={:.5f}, '+'dlnz={:.5f}, '+'dlnz2={:.5f} ')\
#         .format(K, lnz.item(), -dlnz.item(), dlnz2.item()*beta.item()**2)
# print(message)




'''
betaval = beta.item()
##### Compare with exact results (thermodynamic limit)
maglambda = 1 / (np.sinh(2 / Tval) ** 2)
N = 1000000;
x = np.linspace(0, np.pi, N + 1)
y = np.log(np.cosh(2 * betaval) * np.cosh(2 * betaval) + (1 / maglambda) * np.sqrt(
        1 + maglambda ** 2 - 2 * maglambda * np.cos(2 * x)))
FreeExact = -Tval * ((np.log(2) / 2) + 0.25 * sum(y[1:(N + 1)] + y[:N]) / N)
RelFreeErr = abs((FreeEnergy.item() - FreeExact) / FreeExact)
print('FreeEnergy = ', FreeEnergy.item())
print('Exact = ', FreeExact)
print('RelFreeErr = ', RelFreeErr)

'''


'''
filename = 'ising_ctmrg_'+ lattice + '.txt'
with open(filename, 'w') as f:
    # f.write('# correlation_length v.s m \n')
    f.write('# beta \t FreeEnergy \t Energy \t beta^2*dlnz2 \t mag \n')
'''

'''
# for K in np.linspace(0.4, 0.5, 21):
for K in np.linspace(1/1.54, 1/1.49, 21):
    beta = torch.tensor(K, dtype=dtype, device=device).requires_grad_()
    lnz, mag = free_energy_ctmrg(lattice, beta,chiM, CTMRG_step, thres=1e-6)
    dlnz, = torch.autograd.grad(lnz, beta, create_graph=True)  # En = -d lnZ / d beta
    dlnz2, = torch.autograd.grad(dlnz, beta)  # Cv = beta^2 * d^2 lnZ / d beta^2
    # print(K, lnz.item(), -dlnz.item(), dlnz2.item() * beta.item() ** 2)
    # f = open(filename, 'a')
    message = ('{:.6f} ' + 4*'{:.5f} ')\
        .format(K, (-1/K*lnz).item(), -dlnz.item(), dlnz2.item() * beta.item() ** 2, mag)
    print(message)
    # f.write(message)
    # f.write('\n')
    # f.close()
'''

