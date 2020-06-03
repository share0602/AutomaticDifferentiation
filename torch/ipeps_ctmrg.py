import numpy as np
from numpy import linalg as LA
import torch
dtype = torch.float64; device = 'cpu'
torch.autograd.set_detect_anomaly(True)
from torch_ctmrg_two_sites import ctmrg



class iPEPS(torch.nn.Module):
    def __init__(self, params):
        super(iPEPS, self).__init__()

        self.chi = params[0]
        self.hloc = params[1]
        self.ctm_step = params[2]
        # B(phy, up, left, down, right)
        if model_name == 'Heisenberg':
            Init = np.load("HeisenInit.npy")
        elif model_name == 'Ising':
            Init = np.load("IsingInit.npy")
        Ainit = Init[0]; Binit = Init[1]
        Ainit = torch.from_numpy(Ainit); Ainit.requires_grad_()
        Binit = torch.from_numpy(Binit); Binit.requires_grad_()
        # Ainit = torch.rand(2, 2, 2, 2, 2, dtype=torch.float64, device='cpu')
        # Binit = torch.rand(2, 2, 2, 2, 2, dtype=torch.float64, device='cpu')
        self.A = torch.nn.Parameter(Ainit)
        self.B = torch.nn.Parameter(Binit)

    def forward(self):
        chi,hloc, ctm_step = self.chi, self.hloc, self.ctm_step
        # A, B = self.A, self.B
        A = self.A
        # print(A.norm())
        A = A/A.norm();
        #B = B/A.norm()
        # print(A.norm())
        loss = ctmrg(A,A,hloc,chi,ctm_step)
        return loss

# %%%%% Set bond dimension and lattie sites
chi = 16;  # bond dimension
ctm_step = 5
# RGstep = 2*log2L-2; # remain 4 tensors
model_name = 'Ising'
# %%%%% Initialize Hamiltonian (Heisenberg Model) %%%%%
sX = np.array([[0, 1], [1, 0]]);
sY = np.array([[0, -1j], [1j, 0]])
sZ = np.array([[1, 0], [0, -1]]);
sI = np.eye(2)
Init = {};
if model_name == 'Heisenberg':
    ##### Example 1: Heisenberg model
    EnExact = -0.6694421
    hloc = 0.25 * np.real(np.kron(sX, sX) + np.kron(sY, sY) + np.kron(sZ, sZ))
elif model_name == 'Ising':
    ##### Example 2: Ising model
    EnExact = -3.28471
    hmag = 3.1
    hloc = np.real(-np.kron(sX, sX) - hmag * 0.25 * (np.kron(sI, sZ) + np.kron(sZ, sI)))

hloc = hloc.reshape([2]*4)
hloc = torch.from_numpy(hloc)
params_fix = [chi, hloc, ctm_step]
model = iPEPS(params_fix)
optimizer = torch.optim.LBFGS(model.parameters(), max_iter=10)
# optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
params = list(model.parameters())
params = list(filter(lambda p: p.requires_grad, params))
nparams = sum([np.prod(p.size()) for p in params])
print('model:'+model_name)
print('total nubmer of trainable parameters:', nparams)
print(('other parameters: chi:{:d}, ctmStep:{:d}').format(chi, ctm_step))

def closure():
    # print('closure is executed')
    optimizer.zero_grad()
    loss = model.forward()
    loss.backward()
    # print(loss.item(), model.A.grad[0,0,0,0,:])
    return loss

Nepochs = 100
for epoch in range(Nepochs):
    loss = optimizer.step(closure)
    with torch.no_grad():
        # print('torch.no_grad is executed')
        En = model.forward()
        message = ('epoch:{}, ' + 1 * 'En:{:.8f} ').format(epoch, En)
        print(message)
