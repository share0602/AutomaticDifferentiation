import torch
from torch.utils.checkpoint import checkpoint
import sys
sys.path.insert(0, '../')
from utils import ncon
from utils import QR
from utils import SVD
from utils import EigenSolver
torch_qr = QR.apply
torch_svd = SVD.apply
torch_eigh = EigenSolver.apply

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
    u = u[:, :dim_new]
    u_up = u;
    u_down = u
    return u_up, u_down
def coarse_grain(c1t1, t4w, c4t3, u_up, u_down):
    c1 = c1t1 @ u_up
    t4 = ncon([u_down, t4w, u_up],
              [[1, -1], [1, -2, 2], [2, -3]])
    c4 = u_down.T.contiguous() @ c4t3
    return c1, t4, c4
def normalize(x):
    return x/x.norm()
def Renormalize(*tensors):
    cns, tms, T, chi = tensors
    dimT, dimE = T.shape[0], tms[0].shape[0]
    D_new = min(dimE * dimT, chi)
    for _ in range(4):
        c1t1, t4w, c4t3 = extend_tensors(cns[0], tms[0], tms[3], T, cns[3], tms[2])
        u_up, u_down = create_projectors_orus(c1t1, c4t3, D_new)
        cns[0], tms[3], cns[3] = coarse_grain(c1t1, t4w, c4t3, u_up, u_down)
        cns, tms = map(list_rotation, (cns, tms))
        T = weight_rotate(T)
    cns = list(map(normalize, cns))
    tms = list(map(normalize, tms));
    C = (cns[0] + cns[1] + cns[2] + cns[3])/4
    E = (tms[0] + tms[1] + tms[2] + tms[3])/4
    C = 0.5 * (C + C.t())
    E = 0.5 * (E + E.permute(2, 1, 0))
    return C / C.norm(), E/E.norm()

def CTMRG_reproduce(T, chi, max_iter, use_checkpoint=False):
    C = T.sum((0,1))  #
    E = T.sum(1).permute(0,2,1)
    cns = [C] * 4
    tms = [E]*4
    for n in range(max_iter):
        tensors = cns, tms, T, torch.tensor(chi)
        if use_checkpoint: # use checkpoint to save memory
            cns, tms = checkpoint(Renormalize, *tensors)
        else:
            cns, tms = Renormalize(*tensors)

    # print ('ctmrg converged at iterations %d to %.5e, truncation error: %.5f'%(n, diff, truncation_error/n))
    return cns[0], tms[0]

if __name__=='__main__':
    import time
    torch.manual_seed(42)
    D = 6
    chi = 10
    max_iter = 10
    device = 'cpu'

    # T(u,l,d,r)
    T = torch.randn(D, D, D, D, dtype=torch.float64, device=device, requires_grad=True)

    T = (T + T.permute(0, 3, 2, 1))/2.      # left-right symmetry
    T = (T + T.permute(2, 1, 0, 3))/2.      # up-down symmetry
    T = (T + T.permute(3, 2, 1, 0))/2.      # skew-diagonal symmetry
    T = (T + T.permute(1, 0, 3, 2))/2.      # digonal symmetry
    T = T/T.norm()
    T2 = T.clone()
    # C, E = CTMRG(T, chi, max_iter, use_checkpoint=True)
    # C2, E2 = CTMRG_reproduce(T2, chi, max_iter, use_checkpoint=False)
    # print('diffC = ', torch.dist(C2, C2.t()))
    C, E = CTMRG(T, chi, max_iter, use_checkpoint=False)
    # print(C2[0,:], C[0,:])
    # print(C2.norm(), E2.norm())
    # print(C.norm(), E.norm())
    print((T-T2).norm())
    # print('diffC = ', (C2.t()-C2).norm())
    # print('diffE = ', (E2.permute(2, 1, 0)-E2).norm())
    # print( 'diffC = ', torch.dist(C, C.t()))
    # print( 'diffE = ', torch.dist(E, E.permute(2, 1, 0)))

