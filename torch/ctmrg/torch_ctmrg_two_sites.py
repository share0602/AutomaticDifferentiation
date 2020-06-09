
from utils import ncon
from utils import QR
from utils import SVD
from utils import EigenSolver
torch_svd = SVD.apply
torch_eigh = EigenSolver.apply
torch_qr = QR.apply
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
    m1 = c1t1.T @ c1t1;
    m2 = c4t3 @ c4t3.T;
    w, u = torch_eigh(m1 + m2)
    u = torch.flip(u, [1])
    u = u[:, :dim_new]
    u_up = u;
    u_down = u
    return u_up, u_down

def double_ten(ten_a):
    double_ten = ncon([ten_a, ten_a],
                     [[-1,-3,-5,-7,1], [-2,-4,-6,-8,1]])
    return double_ten
def normalize(x):
    return x/x.norm()
def weight_cns_tms(double_ten):
    dy, dz = double_ten.shape[:2]
    c1 = double_ten.reshape((dy, dy, dz * dz, dy * dy, dz, dz))
    c1 = torch.einsum('i i j k l l->j k', c1)
    c2 = double_ten.reshape((dy, dy, dz, dz, dy * dy, dz * dz))
    c2 = torch.einsum('i i j j k l->k l', c2)
    c3 = double_ten.reshape((dy * dy, dz, dz, dy, dy, dz * dz))
    c3 = torch.einsum('i j j k k l->l i', c3)
    c4 = double_ten.reshape((dy * dy, dz * dz, dy, dy, dz, dz))
    c4 = torch.einsum('i j k k l l->i j', c4)
    t1 = torch.einsum('i i j k l->j k l', double_ten.reshape((dy, dy, dz * dz, dy * dy, dz * dz)))
    t2 = torch.einsum('i j j k l->k l i', double_ten.reshape((dy * dy, dz, dz, dy * dy, dz * dz)))
    t3 = torch.einsum('i j k k l->l i j', double_ten.reshape((dy * dy, dz * dz, dy, dy, dz * dz)))
    t4 = torch.einsum('i j k l l->i j k', double_ten.reshape((dy * dy, dz * dz, dy * dy, dz, dz)))

    cns = list(map(normalize, (c1, c2, c3, c4)))
    tms = list(map(normalize, (t1, t2, t3, t4)))
    w = double_ten.reshape([dy**2,dy**2,dy**2,dy**2])
    return w, cns, tms

def create_weight_imp(ten_a, ten_b, two_sites_op):
    weight_imp = ncon([ten_a, ten_b, two_sites_op, ten_a, ten_b],
                      [[-1,-3,5,-11,1], [5,-5,-7,-9,2], [1,2,3,4], [-2,-4,6,-12,3], [6,-6,-8,-10,4]])
    dim = ten_a.shape[0]**2
    weight_imp = weight_imp.reshape([dim]*6)
    return weight_imp


def corner_big(c1t1_a, t4w_a, c4t3_a):
    dim1 = c1t1_a.shape[0]*t4w_a.shape[1]
    dim2 = t4w_a.shape[2]
    c_a1_big = ncon([c1t1_a, t4w_a],
                    [[-1, 1], [1, -2, -3]]).reshape([dim1, dim2])
    dim1 = t4w_a.shape[0]; dim2 = t4w_a.shape[1]*c4t3_a.shape[1]
    c_a4_big = ncon([c4t3_a, t4w_a],
                    [[1, -3], [-1, -2, 1]]).reshape([dim1, dim2])
    return c_a1_big, c_a4_big
def coarse_grain(c1t1_a, t4w_a, c4t3_a, u_ba, d_ba, u_ab, d_ab):
    c1_b = c1t1_a @ u_ba
    t4_b = ncon([d_ba, t4w_a, u_ab],
              [[1, -1], [1, -2, 2], [2, -3]])
    c4_b = d_ab.T.contiguous() @ c4t3_a
    return c1_b, t4_b, c4_b

def ctmrg(ten_a, ten_b, hloc, dim_cut, ctm_step):
    double_ten_a = double_ten(ten_a)
    double_ten_b = double_ten(ten_b)
    w_a, cns_a, tms_b = weight_cns_tms(double_ten_a)
    w_b, cns_b, tms_a = weight_cns_tms(double_ten_b)
    weight_imp = create_weight_imp(ten_a, ten_b, hloc)

    energy = 0
    energy_mem = -1
    for i in range(ctm_step):
        print('ctm step = ', i+1)
        # print('energy = ', energy.item(), 'ctm step = ', i)
        # print('dE = ', abs(energy - energy_mem).item())
        for _ in range(4):
            c1t1_a, t4w_a, c4t3_a = extend_tensors(cns_a[0], tms_a[0], tms_a[3], w_a, cns_a[3], tms_a[2])
            c1t1_b, t4w_b, c4t3_b = extend_tensors(cns_b[0], tms_b[0], tms_b[3], w_b, cns_b[3], tms_b[2])
            c_a1_big, c_a4_big = corner_big(c1t1_a, t4w_a, c4t3_a)
            c_b1_big, c_b4_big = corner_big(c1t1_b, t4w_b, c4t3_b)
            u_ab, d_ab = create_projectors_orus(c_a1_big, c_b4_big, dim_cut)
            u_ba, d_ba = create_projectors_orus(c_b1_big, c_a4_big, dim_cut)
            cns_b[0], tms_b[3], cns_b[3] = coarse_grain(c1t1_a, t4w_a, c4t3_a, u_ba, d_ba, u_ab, d_ab)
            cns_a[0], tms_a[3], cns_a[3] = coarse_grain(c1t1_b, t4w_b, c4t3_b, u_ab, d_ab, u_ba, d_ba)
            cns_a, tms_a, cns_b, tms_b = map(list_rotation, (cns_a, tms_a, cns_b, tms_b))
            w_a, w_b = map(weight_rotate, (w_a, w_b))
        cns_a = list(map(normalize, cns_a));
        cns_b = list(map(normalize, cns_b))
        tms_a = list(map(normalize, tms_a));
        tms_b = list(map(normalize, tms_b))
    w_ab = ncon([w_a, w_b],
                [[-1, -2, 1, -6], [1, -3, -4, -5]])
    norm = ncon([cns_a[0], tms_a[0], cns_a[1], tms_a[3], tms_a[1], w_ab,
                 tms_b[3], tms_b[2], cns_b[3], tms_b[2], cns_b[2]],
                [[1, 3], [2, 4, 1], [5, 2], [3, 6, 8], [10, 7, 5], [4, 7, 12, 14, 11, 6],
                 [8, 11, 13], [15, 12, 10], [13, 16], [16, 14, 17], [17, 15]])

    expect = ncon([cns_a[0], tms_a[0], cns_a[1], tms_a[3], tms_a[1], weight_imp,
                   tms_b[3], tms_b[2], cns_b[3], tms_b[2], cns_b[2]],
                  [[1, 3], [2, 4, 1], [5, 2], [3, 6, 8], [10, 7, 5], [4, 7, 12, 14, 11, 6],
                   [8, 11, 13], [15, 12, 10], [13, 16], [16, 14, 17], [17, 15]])
    energy_mem = energy
    energy = expect / norm * 2
    print('energy = ', energy.item())
    return energy
########################## Main Program ####################################
if __name__ == '__main__':
    model_name = 'Heisenberg'
    dim_cut = 20
    ctm_step = 5
    sX = np.array([[0, 1], [1, 0]])
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0, -1]])
    sI = np.eye(2)
    whichExample = 1
    if model_name == 'Heisenberg':
        ##### Example 1: Heisenberg model
        Init = np.load("../InitialState/HeisenInit.npy")  # obtain from iPEPS-TEBD
        EnExact = -0.6694421
        hloc = 0.25 * np.real(np.kron(sX, sX) + np.kron(sY, sY) + np.kron(sZ, sZ))
    elif model_name == 'Ising':
        ##### Example 2: Ising model
        Init = np.load("../InitialState/IsingInit.npy")  # obtain from iPEPS-TEBD
        EnExact = -3.28471
        hmag = 3.1
        hloc = np.real(-np.kron(sX, sX) - hmag * 0.25 * (np.kron(sI, sZ) + np.kron(sZ, sI)))
    hloc = hloc.reshape([2]*4)
    hloc = torch.from_numpy(hloc)
    ten_a = Init[0]; ten_b = Init[1]
    ten_a = torch.from_numpy(ten_a); ten_a.requires_grad_()
    ten_b = torch.from_numpy(ten_b); ten_b.requires_grad_()
    energy = ctmrg(ten_a, ten_b, hloc, dim_cut, ctm_step)
    print('Final = ',energy)
    # energy.backward()
    # print('ten_a.grad.shape = ', ten_a.grad)
