import torch
import numpy as np

import config
import PSD_ops_torch

from PSD_ops_torch import Tensor
from tqdm import trange

## computation of the Wasserstein2/Gelbrich
## distance between Gaussian distributions

def gb2_par_torch(mlis):
    device = mlis[4]

    x1 =   Tensor( mlis[2] ).to(device)
    mx = torch.mean (x1,axis = 0)
    Sx = torch.cov(x1.T) 

    return PSD_ops_torch.Gelbrich2( mlis[0].to(device), Sx, mlis[1].to(device), mx).item() / mlis[3]
    
def make_Px(T,A,Q,P0):
    
    Pxs = [ P0 ]
    for _ in trange(1,T,desc = 'make_Px'):
        Pxs.append( PSD_ops_torch.bli(A,Pxs[-1]) + Q )
    return Pxs
    
def make_ref_distribution(T,A,Q,P0):
    
    nx = A.size(0)
    mx = torch.zeros(nx*T, dtype=config.conf_prec_type_torch)
    
    if T == 1:
        return P0,mx, A 
    
    Px1, mx1, Is = make_ref_distribution(T-1, A,Q, PSD_ops_torch.bli(A,P0)+Q )
    
    S1 = P0@Is.T
    top = torch.hstack([P0,   S1  ])
    bot = torch.hstack([S1.T, Px1 ])
    
    return torch.vstack([top, bot]), mx, torch.vstack([A, Is@A])
    
def compute_statistics(T,A,Q,P0,TplotP,TplotDT):
    Sxx = dict()
    Mxx = dict()
    Gbxref = []
    gbTs = []
    
    Px = make_Px(T,A,Q,P0)
    for t in trange(1,T+1,TplotDT, desc='compute_statistics ref'):
       
        Tbase = np.max([0,t-TplotP])

        Sx, mx, _ = make_ref_distribution(t-Tbase,A,Q,Px[Tbase])
        gbTs.append(t-1)
        
        Sxx[t] = Sx
        Mxx[t] = mx
    
    gbTs = np.vstack(gbTs)       
    return gbTs,Sxx,Mxx
  
def compute_perception(x,T,Sxx,Mxx,TplotP,TplotDT):      
    if torch.cuda.is_available():
        gpu_n = torch.cuda.device_count()
        devices = [torch.device("cuda:"+str( t%gpu_n ) ) for t in range(8) ]
    else:
        device = torch.device("cpu")
        devices = [torch.device("cpu") for _ in range(8) ]
    
    bsize = x.size(0)
    Gbxref = []
    for t in trange(1,T+1,TplotDT, desc='estimating Wass. distance'):
        Tbase = np.max([0,t-TplotP])
        x1 = np.reshape(x[:,Tbase:t,:], (bsize,-1))

        Sx = Sxx[t]
        mx = Mxx[t]
        mm = [Sx,mx,x1,t-Tbase,devices[0]]
        Gbxref.append(gb2_par_torch(mm))
        
    Gbxref = np.vstack(Gbxref)
    return Gbxref    