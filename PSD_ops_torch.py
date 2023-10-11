import torch as np
import torch
from torch.autograd import Variable

# from torch import FloatTensor as Tensor
from torch import DoubleTensor as Tensor

#A bool that controls where TensorFloat-32 tensor cores may be used in cuDNN convolutions on Ampere or newer GPUs. See TensorFloat-32(TF32) on Ampere devices.
torch.backends.cudnn.allow_tf32     = False
torch.backends.cudnn.deterministic  = True

from tqdm import trange

import config
g_th  = config.conf_th
g_eps = config.conf_eps

id = lambda x: x
xsy = lambda x: ( x + x.T )/2.
aat = lambda a:  a@a.T

assoc_ = lambda u,d: aat(u@np.diag(d)) 
assoc = lambda u,d: assoc_(u,np.sqrt(d)) 

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")

ZT  = torch.zeros([1]).to(device)


def eig(s,eps = 0e-8,th = g_th):
    d,u = np.linalg.eig(s)
    return d

def trsq(s,th = g_th):
    d,u = np.linalg.eigh(s)
    d  = np.where(d > 1e-21, d, 0.)
    return np.sum( np.sqrt(d) )

def bli(A,B):
    # not always perfectly symmetric, might cause numerics
    return A@B@A.T
    
def sqrtm(s,eps = g_eps,th = g_th):
        
    try:
        d,u = np.linalg.eigh(s)
    except: #torch eigh wont take 0 matrix
         d,u = np.linalg.eig(s)   
         u   = np.real(u)
         d   = np.real(d)
         if not sqrtm.eig_warn:
            print("sqrtmTorchWarning: eig used")
            sqrtm.eig_warn = True
         
    d  = np.where(np.abs(d) > th, d, 0.)
    lmax1 = np.max( np.abs(d) )

    th = eps*lmax1
    assert np.all(d >= -th), "sqrtm error: not PSD (%E,%E)" % (np.max(d),np.min(d))

    d_ = np.where(d > g_th, np.sqrt(d).double(),  0.)
    return assoc(u.double(),d_)

sqrtm.eig_warn = False

def pinv(s,eps = g_eps,th = g_th):
    p = torch.linalg.pinv(s, atol=th, rtol=eps, hermitian=True, out=None)
    return p
    
spinv = lambda x: PSD( sqrtm( pinv(x) ) )
proj = lambda S: S@pinv(S)

def Tstar(s2,s1, eps = 0e-8):
    B = torch.eye(s1.size(0)).to(s1.device)
    _,PIan = Sopt(s1,s2,B,lqmax=lmax(s1).item(),lbmax=1.)
    return PIan
    
def SoptTheorem(Q,M,B,lqmax,lbmax,eps=0e-8):

    B = B / lbmax
    Q = Q / lqmax
    M = M / lqmax

    BMB = bli(B,M)
    BMBsq = sqrtm(BMB)

    S = bli( BMBsq, spinv( bli( BMBsq , Q ) ) )

    PI = Q@S@pinv(BMB)@B@M@pinv(M)

    # # sanity check
    # sqrtm(Q-bli(PI , M))

    return S,PI

def Sopt_Remark(Q,M,B,lqmax,lbmax,eps=0e-8):
    B = B / lbmax
    Bi = np.linalg.inv(B)

    b  = sqrtm(B)
    bi = pinv(b)
    
    Q = Q / lqmax
    M = M / lqmax

    Bm = M@pinv(M)@B
    bMb = bli(b,M)
    bQb = bli(b,Q)

    bMbsq = sqrtm(bMb)
    bQbsq = sqrtm(bQb)
    bQbsqi = spinv(bQb)

    S = bli(bMbsq , spinv( bli(bMbsq , bQb) ) )
    
    bPIbi1 = bQb@S@pinv(bMb)#@bMb@pinv(bMb)
    bPIbi = bQbsq@sqrtm( bli(bQbsq,bMb) )@bQbsqi@pinv(bMb)

    PI = bi@bPIbi@b#@pinv(M)
    
    PI = bi@bPIbi1@bi@M@pinv(M)
    PI = Q@b@S@pinv(bMb)@b@M@pinv(M)

    PI = PI

    return S,PI

Sopt = SoptTheorem
# Sopt = Sopt_Remark

def PSD(s,eps = g_eps,th = g_th):
    s = (s+s.T)/2. # avoid numerics due to asymmetry

    try:
        d,u = np.linalg.eigh(s)
    except: #torch eigh wont take 0 matrix
         d,u = np.linalg.eig(s)   
         u   = np.real(u)
         d   = np.real(d)

    d  = np.where(np.abs(d) > th, d,  torch.zeros([1]).to(d.device))
    lmax = np.max( np.abs(d) )

    th = eps*lmax
    assert np.all(d >= -th), "PSD error: not PSD (%E,%E)" % (np.max(d),np.min(d))

    d  = np.where(d > g_th, d,  torch.zeros([1]).to(d.device))# + eps

    r = assoc(u,d)
    return (r+r.T).double()/2


def lmax(s):
    d,u = np.linalg.eig(s)
    lmax = np.max( np.abs(d) )
    return lmax

def lmin(s):
    d,u = np.linalg.eigh(s)
    lmin = np.min( d )
    return lmin

def lmin_eig(s):
    # return lmin( s )
    d,u = np.linalg.eig(s)
    lmin_ = np.min( np.real( d ) )
    return lmin_
    
def Gelbrich2(s1,s2,m1=0,m2=0, eps = 0e-8):
    
    device = s1.device
    if isinstance(m1,int):
        m1 = Tensor(m1).to(device)
    if isinstance(m2,int):
        m2 = Tensor(m2).to(device)

    l1 = lmax(s1)
    l2 = lmax(s2)

    g = s1 + s2

    trsq1 = trsq( bli(sqrtm(s1), s2) )
    trsq2 = trsq( bli(sqrtm(s2), s1) )

    m = torch.sum((m1-m2)**2)
    return m + torch.trace( g ) - trsq1 -trsq2

    
def Swk_(PXstar,PX,Pi):
    return sqrtm(PX - bli(Pi,PXstar))
    
def sopt_par_(mlist):

    Q = mlist[0]
    M = mlist[1]
    B = mlist[2]
    lqmax = mlist[3]

    _,PIan = Sopt(Q,M,B,lqmax,lbmax=1.)

    return PIan

def sopt_par(Q,P0,Ms,Bs,devices,multi_pool):

    import torch.multiprocessing as mp
    torch.autograd.set_detect_anomaly(True)

    T  = len(Ms)
    lq = lmax(Q)

    mlis = [ [Q.to(devices[t]), Tensor(Ms[t]).to(devices[t]), Bs[t].to(devices[t]), lq.to(devices[t])] for t in range(T)]
    mlis[0][0] = Tensor(P0).to(devices[0])
    mlis[0][3] = lmax( mlis[0][0] )
   
    try:
        
        top_x_ = multi_pool.imap(sopt_par_,mlis)
        PIs = list(top_x_)

    except (KeyboardInterrupt, SystemExit):
        multi_pool.close()
        multi_pool.join()
        raise
    except:
        multi_pool.close()
        multi_pool.join()
        raise

    return PIs