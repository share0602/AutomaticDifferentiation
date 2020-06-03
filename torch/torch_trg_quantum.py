from numpy import linalg as LA
from utils import ncon
from utils import SVD

torch_svd = SVD.apply
import numpy as np
import torch
dtype = torch.float64; device = 'cpu'
torch.autograd.set_detect_anomaly(True)

########################## Utility Function ####################################
def double_ten(Ainit):
    chiD = Ainit.shape[0]
    gA = ncon([Ainit, (Ainit)], [[-1, -3, -5, -7, 1], [-2, -4, -6, -8, 1]]);
    gA = gA.reshape(chiD ** 2, chiD ** 2, chiD ** 2, chiD ** 2);
    return gA
def double_imp_left(Ainit, hl):
    chiD = Ainit.shape[0]
    chiH = hl.shape[2]
    gOA = ncon([Ainit, hl, (Ainit)], [[-1, -3, -5, -8, 1], [1, 2, -7], [-2, -4, -6, -9, 2]]);
    gOA = torch.reshape(gOA, [chiD ** 2, chiD ** 2, chiD ** 2 * chiH, chiD ** 2]);
    return gOA
def double_imp_right(Binit, hr):
    chiD = Binit.shape[0]
    chiH = hr.shape[2]
    gOB = ncon([Binit,hr,(Binit)], [[-1,-4,-6,-8,1], [1,2,-3], [-2,-5,-7,-9,2]]);
    gOB = torch.reshape(gOB, [chiD ** 2 * chiH, chiD ** 2, chiD ** 2, chiD ** 2]);
    return gOB
def construct_uv(TA, chi):
    chi0 = TA.shape[0]
    chi1 = min(chi, chi0 ** 2);
    TA = torch.reshape(TA, (chi0 ** 2, chi0 ** 2))
    utemp, stemp, vtemp = torch_svd(TA)
    vtemp = vtemp.T
    U1 = utemp[:, :chi1] @ torch.diag(torch.sqrt(abs(stemp[:chi1])))
    U1 = torch.reshape(U1, (chi0, chi0, chi1))
    V1 = torch.diag(torch.sqrt(abs(stemp[:chi1]))) @ vtemp[:chi1, :]
    V1 = torch.reshape(V1, (chi1, chi0, chi0))
    V1 = V1.permute([1, 2, 0])
    return U1, V1

def construct_uv_imp_left(gOA, chiH, chi):
    chi2 = gOA.shape[0]
    chi3 = min(chi, chi2 ** 2)
    gOA = torch.reshape(gOA, (chi2 ** 2, chi2 ** 2 * chiH))
    utemp, stemp, vtemp = torch_svd(gOA)
    vtemp = vtemp.T
    U11 = utemp[:, :chi3] @ torch.diag(torch.sqrt(abs(stemp[:chi3])))
    U11 = torch.reshape(U11, (chi2, chi2, chi3))
    V11 = torch.diag(torch.sqrt(abs(stemp[:chi3]))) @ vtemp[:chi3, :]
    V11 = torch.reshape(V11, (chi3, chi2 * chiH, chi2))
    V11 = V11.permute([1, 2, 0])
    return U11, V11
def construct_uv_imp_right(gOB, chiH, chi):
    chi2 = gOB.shape[0]
    chi3 = min(chi, chi2 ** 2)
    gOB = torch.reshape(gOB, (chi2 ** 2 * chiH, chi2 ** 2))
    utemp, stemp, vtemp = torch_svd(gOB)
    vtemp = vtemp.T
    U12 = utemp[:, :chi3] @ torch.diag(torch.sqrt(abs(stemp[:chi3])))
    U12 = torch.reshape(U12, (chi2, chi2 * chiH, chi3))
    V12 = torch.diag(torch.sqrt(abs(stemp[:chi3]))) @ vtemp[:chi3, :]
    V12 = torch.reshape(V12, (chi3, chi2, chi2))
    V12 = V12.permute([1, 2, 0])
    return U12, V12

def constract_four_ten(V1,U2,V2,U1):
    return ncon([V1, U2, V2, U1], [[1, 4, -1], [2, 1, -2], [4, 3, -4], [3, 2, -3]]);


########################## Main Function ####################################
def trg_loss(Ainit, Binit, chi, log2L, hl, hr):
    # print('trg_loss is executed')
    chiD = Ainit.shape[0]
    chiH = hl.shape[2]
    RGstep = 2 * log2L - 2;  # remain 4 tensors
    # %%%%% Construct <psi| psi> & <psi|H| psi>
    gA = double_ten(Ainit)
    gB = double_ten(Binit)
    gOA = double_imp_left(Ainit, hl)
    gOB = double_imp_right(Binit, hr)
    TA = gA.clone(); TB = gB.clone();
    Nsq = 1;
    checkConv_norm = []
    checkConv_expect = []
    # %%%%% do TRG iteration
    for iter in range(RGstep):
         print('RGstep = ',iter+1, )
         Tnorm = torch.sqrt(torch.norm(TA) * torch.norm(TB));
         Nsq = Nsq * torch.exp(torch.log(Tnorm) / (2 ** (iter)));
         TA = TA/Tnorm; TB = TB/Tnorm;
         U1, V1 = construct_uv(TA, chi)
         TB = TB.permute(3,0,1,2)
         U2, V2 = construct_uv(TB, chi)
         #%%%%% Now we are going to consider the expectation value! %%%%%
         gOA, gOB, gA, gB = map(lambda x: x/Tnorm,(gOA, gOB, gA, gB) )
         # decompose gOA and gOB
         if iter == 0: # incoporate hl,hr
             U11, V11 = construct_uv_imp_left(gOA, chiH, chi)
             gOB = gOB.permute(3,0,1,2)
             U12, V12 = construct_uv_imp_right(gOB, chiH, chi)
         else:
             U11, V11 = construct_uv(gOA, chi)
             gOB = gOB.permute(3, 0, 1, 2)
             U12 ,V12 = construct_uv(gOB, chi)

         gB = gB.permute(3, 0, 1, 2)
         U21, V21 = construct_uv(gB, chi)
         U22, V22 = construct_uv(gA, chi)

         # construct new gOA,gOB,gB,gA
         TA = constract_four_ten(V1, U2, V2, U1)
         TB = constract_four_ten(V1, U2, V2, U1)
         gOB = constract_four_ten(V11, U12, V2, U1)
         gA = constract_four_ten(V22, U2, V12, U1)
         gOA = constract_four_ten(V1, U21, V2, U11)
         gB = constract_four_ten(V1, U2, V21, U22)

         checkConv_expect.append(torch.norm(gOA).item())
         checkConv_norm.append(Tnorm.item())

    #  contract the remaining 4 tensor
    TATA = ncon([TA,TA],[[1,-1,2,-3],[2,-2,1,-4]]);
    Rest = ncon([TATA,TATA],[[1,2,3,4],[3,4,1,2]]);
    Nsq = Nsq * torch.exp(torch.log(Rest) / 2 ** (2 * log2L));
    T11T12 = ncon([gOA,gOB],[[1,-1,2,-3], [2,-2,1,-4]]);
    T21T22 = ncon([gB,gA],[[1,-1,2,-3], [2,-2,1,-4]]);
    ThamRest = ncon([T11T12,T21T22],[[1,2,3,4],[3,4,1,2]]);
    ExpectEner = 2 * ThamRest / Rest; # we only evaluate 2 interaction for each sites
    print('ExpectEner.item() = ', ExpectEner.item())
    return ExpectEner

########################## Main Program ####################################
if __name__ == '__main__':
    # %%%%% Set bond dimension and lattie sites
    log2L = 5; #2^10 x 2^10 site
    chi = 16; # bond dimension
    # RGstep = 2*log2L-2; # remain 4 tensors
    whichExample = 2
    # %%%%% Initialize Hamiltonian (Heisenberg Model) %%%%%
    sX = np.array([[0, 1], [1, 0]]);
    sY = np.array([[0, -1j], [1j, 0]])
    sZ = np.array([[1, 0], [0, -1]]);
    sI = np.eye(2)
    Init = {};
    if whichExample == 1:
        ##### Example 1: Heisenberg model
        Init = np.load("HeisenInit.npy") # obtain from iPEPS-TEBD
        EnExact = -0.6694421
        hloc = 0.25 * np.real(np.kron(sX, sX) + np.kron(sY, sY) + np.kron(sZ, sZ))
    elif whichExample == 2:
        ##### Example 2: Ising model
        Init = np.load("IsingInit.npy") # obtain from iPEPS-TEBD
        EnExact = -3.28471
        hmag = 3.1
        hloc = np.real(-np.kron(sX, sX) - hmag * 0.25 * (np.kron(sI, sZ) + np.kron(sZ, sI)))


    uF, sF, vhF = LA.svd(hloc.reshape(2, 2, 2, 2).transpose(0, 2, 1, 3).reshape(4, 4))
    chicut = sum(sF > 1e-12)
    hl = (uF[:, :chicut] @ np.diag(np.sqrt(abs(sF[:chicut])))).reshape(2, 2, chicut)
    hr = ((vhF.T[:, :chicut]) @ np.diag(np.sqrt(abs(sF[:chicut])))).reshape(2, 2, chicut)
    chiH = hl.shape[2]

    hl = torch.from_numpy(hl)
    # print(hl)
    # hl = hl.type(torch.DoubleTensor)
    hr = torch.from_numpy(hr)
    # print(hl.dtype)

    Ainit = Init[0]; Binit = Init[1]
    Ainit = torch.from_numpy(Ainit)
    Ainit.requires_grad_()
    Binit = torch.from_numpy(Binit)
    Binit.requires_grad_()


    ExpectEner = trg_loss(Ainit, Binit, chi, log2L, hl, hr)
    ExpectEner.backward()
    print('Ainit.grad.shape = ', Ainit.grad.shape)
    print('Ainit.grad[0,0,0,0,:] = ', Ainit.grad[0,0,0,0,:])
    print('Ainit = ', Ainit)
    ExpectEner = ExpectEner.item()
    Error = abs((ExpectEner - EnExact)/EnExact);
    # print('NormPerSite = ',Nsq);
    print('ExpectEner = ', ExpectEner, ', EnExact = ', EnExact)
    print('Error = ', Error)
    # print(checkConv_norm)
    # print(checkConv_expect)