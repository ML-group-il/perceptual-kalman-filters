##########################
# Perceptual Kalman Filters for pytorch
# by 
# ~
# Implementation of PKF algorithm and related code from "PKF: online state estimation under a perfect perceptual quality constraint" (2023) by anonymous author(s)
########################################

import numpy as np
import scipy
import torch
from torch.autograd import Variable
# from torch import FloatTensor as Tensor
from torch import DoubleTensor as Tensor

torch.backends.cudnn.allow_tf32 = False
# A bool that controls where TensorFloat-32 tensor cores may be used in cuDNN convolutions on Ampere or newer GPUs. See TensorFloat-32(TF32) on Ampere devices.

torch.backends.cudnn.deterministic=True

torch.set_default_tensor_type(torch.DoubleTensor)


from tqdm import tqdm,trange

import config
import PSD_ops_torch
prec_type = config.conf_prec_type

## utils
def sy2(a):
    return a + a.T

def mat_quad_form(X,P):
    return X@(P@X.T)

def qf(X,P):
    return mat_quad_form(X,P)        

## Theoretic MSE of ``PKF'' filter
def thMSE(A,Q,P0,Pis,SKM,Ps):
    mses = []

    Ms = SKM[-1]
    D = 0.*Q
    Qt = P0

    for t in trange(0,len(Pis), desc = 'thMSE'):

        D = A@D@A.T + Qt + Ms[t] - Pis[t]@Ms[t] - (Pis[t]@Ms[t]).T
        mses.append( torch.trace(Ps[t]+D))
        Qt = Q

    return mses

## Theoretic MSE of ``tic'' filter
def lboundMSE(A,Q,P0,SKM,Ps, mp = None):

    Ms = SKM[-1]
    T = len(Ms)

    PXs = []
    PXstars = []
    Gs = []
    MSE = []

    PXs.append(P0)
    PXstars.append(Ms[0])

    for k in trange(1,T, desc = 'lboundMSE'):
        
        PXs.append( PSD_ops_torch.bli(A,PXs[-1]) + Q )
        PXstars.append( PSD_ops_torch.bli(A,PXstars[-1]) + Ms[k])
        
    try:
        
        parmap = mp.starmap(PSD_ops_torch.Gelbrich2,zip(PXs,PXstars))
        Gs = list(parmap)
        MSE = [ torch.trace(Ps[k])+Gs[k] for k in range(T) ]

    except (KeyboardInterrupt, SystemExit):
        mp.close()
        mp.join()
        raise
    except:
        mp.close()
        mp.join()
        raise

    return (MSE, Gs, PXs, PXstars)


## standard Kalman Filter for pytorch
def kalman(yin,A,C,Q,R,P0, C0=None, R0=None, eps=0.e-2, output_innovation_cov=False, mp=None):

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        
    else:
        device = torch.device("cpu")
        
    yin = yin.to(device)
    A   = A.to(device)
    Q   = Q.to(device)
    P0  = P0.to(device)
        
    batch_size, T, ny = yin.size()
    
    if not isinstance(R,list):
        R  = R.to(device)
        R_ = [R for _ in range(1)]
    else:
        R_ = R.to(device)
        

    if not isinstance(C,list):
        C = [C.to(device) for _ in range(T)]
    else:
        C = [c.to(device) for c in C]

    if C0 is None:
        C0 = C[0]
    else:
        C0   = C0.to(device)


    if R0 is None:
        R0 = R_[0]
    else:
        R0   = R0.to(device)
        
    ny, nx = C0.size()
    Inx = torch.eye(nx).to(device)

    Ss = []
    Ks = []
    Ms = []

    Ps = []

    shape_tup = [batch_size,T,nx]
    xout = torch.zeros(shape_tup, dtype = config.conf_prec_type_torch).to(device)
    Iout = torch.zeros([batch_size,T,ny], dtype = config.conf_prec_type_torch).to(device)

    Sy0 = PSD_ops_torch.bli(C0,P0) + R0
    Sstr0 = PSD_ops_torch.bli( P0@C0.T, PSD_ops_torch.pinv(Sy0) )

    P = P0
    K = P0@C0.T@PSD_ops_torch.pinv(Sy0)
    S = Sy0
    M = PSD_ops_torch.bli(K,S)
    
    if not output_innovation_cov:
        Ss.append( None )
    else:
        Ss.append( S )
    Ks.append( K )
    Ms.append( M )

    P = P0 - Sstr0
    Ps.append( P )

    del C0, R0

    for t in trange(1,T, desc = 'kalman matrices'):
        
        Rt = R
        
        P_  = PSD_ops_torch.bli(A,P)   +  Q
        S   = PSD_ops_torch.bli(C[t],P_) + Rt
        K   = P_@C[t].T@ PSD_ops_torch.pinv(S)

        M = PSD_ops_torch.bli(K,S)
        
        if not output_innovation_cov:
            Ss.append( None )
        else:
            Ss.append( S )
        Ks.append( K )
        Ms.append( M )

        P = (Inx - K@C[t])@P_
        Ps.append( P )

    y0 = yin[:,0,:]
    Iout[:,0,:] = y0

    x = Ks[0]@y0.T
    xout[:,0,:] = x.T

    for t in trange(1,T, desc='kalman filtering'):

        y = yin[:,t,:]
        x_ = A@x
        I = y.T - C[t]@x_
        Iout[:,t,:] = I.T

        x = x_ + Ks[t]@I
        xout[:,t,:] = x.T

    
    SKM_ = (Ss,Ks,Ms, Ps)
    SKM = [ [t.cpu()  if t is not None else t for t in s] for s in SKM_]
    return xout.cpu(), Iout.cpu(), SKM[:-1], SKM[-1]

## computation of PKF coefficients
def pkal_matrices(T,A,Q,P0,SKM, weights = None, eps=0e-2, multi_pool = None, isGreedy = False, devices_in = None):

    if weights is None:
        weights = torch.ones(T)#/(T+1)

    elif isinstance(weights, int):
        ind = weights
        weights = torch.zeros(T)
        weights[ind] = 1.

    PIs = []
    Ds  = []
    TRs = []
    Const = []

    Piv = []
    Div = []

    Ms = SKM[-1]

    nx, _ = A.size()
    Inx = torch.eye(nx)

    lqmax = PSD_ops_torch.lmax(Q)

    if torch.cuda.is_available():
        
        device = torch.device("cpu")

        gpu_n = torch.cuda.device_count()
        
        devices = [torch.device("cuda:"+str( t%gpu_n ) ) for t in range(T) ]

    else:
        device = torch.device("cpu")
        devices = [torch.device("cpu") for _ in range(T) ]
        
    if devices_in is not None:
        devices = devices_in 

    Qtorch = Tensor(Q).to(device)
    Atorch = Tensor(A).to(device)

    tInx = torch.eye(nx).to(device)
    Asum = 0.*tInx
    Anf = 1.

    Asms = []
    nrBs = []

    for k in trange(T-1,0,-1):

        M  = Ms[k]
        Asum  = PSD_ops_torch.bli(Atorch.T,Asum) + weights[k]*tInx
        Asum_in = Asum / (PSD_ops_torch.lmax(Asum).item() + 1e-11)
        
        Asms.append(Asum.cpu().detach().numpy())
        nrBs.append(Asum_in)

    M  = Ms[0]

    Asum  = PSD_ops_torch.bli(Atorch.T,Asum) + weights[0]*tInx#/Anf
    Anf *= PSD_ops_torch.lmax(Asum).item()
    Asum_in = Asum / (PSD_ops_torch.lmax(Asum).item() +1e-11)

    M  = Ms[0]

    Asms.append(Asum.cpu().detach().numpy())
    nrBs.append(Asum_in)

    Asms = Asms[::-1]
    nrBs = nrBs[::-1]
    
    if isGreedy:
        nrBs = [torch.eye(nx) for _ in nrBs]
    
    if not multi_pool is None:
        PIpar = PSD_ops_torch.sopt_par(Qtorch,P0,Ms,nrBs,devices, multi_pool=multi_pool)
        Piv = [pi.cpu() for pi in PIpar]
    else:
        Piv.append(
            PSD_ops_torch.Sopt(Tensor(P0).to(device), Tensor(Ms[0]).to(device),nrBs[0],lqmax=PSD_ops_torch.lmax(P0),lbmax=1.)[1]
            )

        Piv = Piv + [ PSD_ops_torch.Sopt(Qtorch, Tensor(Ms[t]).to(device),nrBs[t],lqmax=lqmax,lbmax=1.)[1].cpu() for t in range(1,T) ]


    D = P0 + Ms[0] - Ms[0]@Piv[0].T - Piv[0]@Ms[0]
    Div.append(D)
    for t in range(1,T):
        Div.append(A@Div[t-1]@A.T + Q + Ms[t] - Ms[t]@Piv[t].T - Piv[t]@Ms[t])

    torch.cuda.empty_cache()
    return (Piv,Div,Asms )

def pkal_par(m):
    return pkal(m[0],m[1],m[2],m[3],m[4],pis=m[5])[0]
    
## Perceptual Kalman filtering (PKF)
def pkal(Iin,A,Q,P0,SKM, pis = None, eps=0.e-2):
    
    batch_size, T, ny = Iin.shape
    nx, _ = A.shape
    Inx = np.eye(nx)

    Ss = SKM[0]
    Ks = SKM[1]
    Ms = SKM[2]

    shape_tup = [batch_size,T,nx]
    Xout = torch.zeros(shape_tup, dtype = config.conf_prec_type_torch)
    wout = 0.

    y0 = Iin[:,0,:]
    Pi = pis[0]

    K = Ks[0]
    
    swk0 = ( P0 - PSD_ops_torch.bli(pis[0],Ms[0]) + 0e-12*Inx )
    Swks_sq0 = PSD_ops_torch.sqrtm(swk0) 
    
    w0 = Swks_sq0@torch.randn(nx,batch_size)
    X = Pi@K@y0.T + w0
    
    Xout[:,0,:] = X.T

    for t in trange(1,len(pis), desc = 'PKF filtering'):
        
        X_ = A@X
        I = Iin[:,t,:]

        K = Ks[t]
        Pi = pis[t]
        
        pmax = torch.max( torch.abs( Pi ))
        if pmax < 1e-12 and pkal.pi_warn == False:
            print("pkal warning: Pi is very small!!")
            pkal.pi_warn = True
        
        swk_t = (Q - PSD_ops_torch.bli(Pi,Ms[t]) + 0e-12*Inx )
        Swks_sq_t = PSD_ops_torch.sqrtm(swk_t)

        wk = Swks_sq_t@torch.randn(nx,batch_size)
        X = X_ + Pi@K@I.T + wk
        
        Xout[:,t,:] = X.T

    return Xout, Swks_sq0

pkal.pi_warn = False

## Temporally inconsistent (``tic'') filter
def  tic_kalman(xstar_in, PXs, PXstars, mp = None, device_in = None):
    # device = torch.device("cuda:0")
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
    else:
        map_location = 'cpu'
        device = torch.device('cpu')
    
    if device_in is not None:
        device = device_in    
    
    batch_size, T, nx = xstar_in.size()
    shape_tup = [batch_size,T,nx]

    Pis     = []
    swks    = []
    
    xstar_in = xstar_in.to(device)
    Xout = torch.zeros(shape_tup, dtype = config.conf_prec_type_torch).to(device)

    for t in trange(T, desc = 'tic filter'):
        
        PXstars_t = PXstars[t].to(device)
        PXs_t = PXs[t].to(device)
        
        Pis_t = PSD_ops_torch.Tstar(PXstars_t,PXs_t)
        swks_t = PSD_ops_torch.Swk_(PXstars_t,PXs_t,Pis_t)

        wk = swks_t@torch.randn(nx,batch_size).to(device)
        X = Pis_t@xstar_in[:,t,:].T + wk

        Xout[:,t,:] = X.T
        
    torch.cuda.empty_cache()

    return Xout.cpu()
       
## optimized filters
def optpkf(Iin, T,A,Q,P0,SKM, Pis,Phis):
   
    
    if torch.cuda.is_available():
        device = torch.device("cuda:0")

        gpu_n = torch.cuda.device_count()
        devices = [torch.device("cuda:"+str( t%gpu_n ) ) for t in range(T) ]
    else:
        device = torch.device("cpu")
        devices = [torch.device("cpu") for _ in range(T) ]
    device = torch.device("cpu")
    
    nx = A.size(0)
    rT = range(T)

    Ks = SKM[1]
    Ms = SKM[-1]
    Ss = SKM[0]
    
    # AiA = torch.linalg.pinv(A)@A
    # Acf = torch.eye(nx).to(device)
    
    AiA = torch.linalg.pinv(A)#@A
    Acf = A.to(device)
    
    # Pis and Phis  were optimized  
    # multiplied by coefficients Kt and A 
    # respectively, to reduce complexity
    Pis =  [ (Tensor(Pis[t])@torch.linalg.pinv(Ks[t]) ).to(device) for t in rT]
    Phis=  [ (Tensor(Phis[t])@AiA).to(device) for t in rT]
    
    Q_ = Q

    A = Tensor( A ).to(device)
    Q = Tensor( Q ).to(device)
    P0= Tensor( P0).to(device)
    Ms = [Tensor( m ).to(device) for m in Ms]
    Ks = [Tensor( k ).to(device) for k in Ks]    
    Ss = [Tensor( s ).to(device) for s in Ss]
    
    pinvQ = PSD_ops_torch.pinv(Q)
    Iin     = Tensor( Iin )     .to(device)
    
    constraint = P0-qf(Pis[0], Ms[0])
    Dk = [ P0 + Ms[0]  - sy2( Pis[0]@Ms[0] ) ]
    
    batch_size, _, ny = Iin.shape 
    Xout = Tensor( np.zeros( (batch_size,T,nx) ) ).to(device)
    ups_out = Tensor( np.zeros( (batch_size,T,nx) ) ).to(device)
    
    wk = torch.randn(batch_size,nx,device = device)
    wk @= PSD_ops_torch.sqrtm(constraint)

    Jk = Iin[:,0,:]@Ks[0].T@Pis[0].T + wk
    Xout[:,0,:] = Jk
        
    rng = trange(1,T,desc = 'PKF (opt) filtering')
        
    ups_k = 0.*ups_out[:,0,:]
    Sigma_upsilon_k = torch.zeros(nx,nx)
        
    for t in rng:
        
        Psi_k_1 = Ks[t-1]@Ss[t-1]@Ks[t-1].T@Pis[t-1].T 
        Psi_k_1 += A@Sigma_upsilon_k@Acf.T@Phis[t-1].T
        
        Sigma_upsilon_k = qf(A,Sigma_upsilon_k) 
        Sigma_upsilon_k += Ms[t-1]
        if t -1 > 0:
            Sigma_upsilon_k -= qf(Psi_k_1,pinvQ)
        else:
            Sigma_upsilon_k -= qf(Psi_k_1,PSD_ops_torch.pinv(P0))

      
        #verify PSD
        PSD_ops_torch.sqrtm(Sigma_upsilon_k)
        
        constraint = Q-qf(Pis[t], Ms[t])-qf(Phis[t]@Acf, Sigma_upsilon_k)
        
        wk = torch.randn(batch_size,nx,device = device)
        wk @= PSD_ops_torch.sqrtm(constraint)
        
        ups_k = ups_k@A.T + Iin[:,t-1,:]@Ks[t-1].T
        if t -1 > 0:
            ups_k -= Jk@pinvQ@Psi_k_1.T
        else:
            ups_k -= Jk@PSD_ops_torch.pinv(P0)@Psi_k_1.T
        
        Jk = ups_k@Acf.T@Phis[t].T + Iin[:,t,:]@Ks[t].T@Pis[t].T + wk 
                
        Xout[:,t,:] = Xout[:,t-1,:]@A.T + Jk
        ups_out[:,t,:] = ups_k #upsilon_k
            
        Dk.append( qf(A,Dk[t-1]) + Q + Ms[t] - sy2( Pis[t]@Ms[t] + Phis[t]@Acf@Sigma_upsilon_k@A.T ) )
        
        
    Div = [torch.trace(d).item() for d in Dk]
    upsout = ups_out
    
    X_out    = Xout.cpu()
    ups_out = ups_out.cpu() 
    
    return X_out, ups_out, Div
