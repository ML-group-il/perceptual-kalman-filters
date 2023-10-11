import numpy as np
import torch
from tqdm import tqdm,trange
from functools import partial
from itertools import repeat

import config
import PSD_ops_torch
from PSD_ops_torch import Tensor

## Methods to sample Linear-Gauss dynamics


def MSE(x,z):

    nrm = torch.linalg.norm((x-z), axis = 2)
    mse = torch.mean(nrm**2,axis = 0)

    return mse
   
def make_dyn_torch(T,A,C,Qs,R,P0,R0=None,C0=None,X0bsize=[None,64,'cpu']):
    
    ny,nx = C[0].size()
    X0, bsize,device = X0bsize

    A = A.to(device)
    Qs =Qs.to(device)
    
    R0=R0.to(device)
    C0 =C0.to(device)
    P0 =P0.to(device)
    
    rt = R.to(device)
    rts= PSD_ops_torch.sqrtm(rt)
    
    x_ = torch.zeros([bsize,T,nx]).to(device)
    y_ = torch.zeros([bsize,T,ny]).to(device)

    if X0 is None:
        x = torch.randn(bsize,nx).to(device)@PSD_ops_torch.sqrtm(P0)
    else:
        x = Tensor(X0).to(device)

    V = torch.randn(bsize, ny).to(device)@PSD_ops_torch.sqrtm(R0)
    y = x@C0.T + V

    x_[:,0,:] = x
    y_[:,0,:] = y

    for t in trange(1,T,desc = 'generate dynamics...'):
        ct = Tensor(C[t]).to(device)
        
        V = torch.randn(bsize, ny).to(device)@rts
        W = torch.randn(bsize, nx).to(device)@Qs

        x = x@A.T + W
        y = x@ct.T + V

        x_[:,t,:] = x
        y_[:,t,:] = y
        
        del ct
        torch.cuda.empty_cache()

    del A,C,Qs ,R ,R0,C0 ,X0
    torch.cuda.empty_cache()
    return (x_,y_)
            
def make_dyn(args,T,A,C,Q,R,P0,bsize = 64,R0=None,C0=None,X0=None, multi_pool=None):

    if not isinstance(R,list):
        R_ = [R for _ in range(1)]
    else:
        R_=R

    if not isinstance(C,list):
        C = [C for _ in range(T)]

    if C0 is None:
        C0 = C[0]

    if R0 is None:
        R0 = R_[0]

    # C[0] = C0 
    # R[0] = R0

    Qs = PSD_ops_torch.sqrtm(Q)

    baseargs = (T,A,C,Qs,R,P0,R0,C0)#,X0,bsize = 64)
    jobsnum = args.workers#*4
    if torch.cuda.is_available():
        # device = torch.device("cuda:0")
        device = torch.device("cpu")

        gpu_n = torch.cuda.device_count()
        devices = [torch.device("cuda:"+str( t%gpu_n ) ) for t in range(jobsnum) ]
    else:
        devices = [torch.device("cpu") for t in range(jobsnum) ]
    
    party = partial(make_dyn_torch, T,A,C,Qs,R,P0,R0,C0)   
     
    if not multi_pool is None:
        mlis = [bsize//args.workers]*jobsnum
        mlis[-1] += bsize - (bsize//jobsnum)*jobsnum
        X0par = []; X0pos = 0
        for i in range(jobsnum):
            if not X0 is None:
                X0par.append(X0[X0pos:(X0pos+mlis[i]),...])
                X0pos += mlis[i]
            else:
                X0par.append(None)
        
        top_x_ = multi_pool.imap(party, zip(X0par,mlis,devices))
        par_res = list(top_x_)
        x = torch.vstack([p[0] for p in par_res])
        y = torch.vstack([p[1] for p in par_res])
    else:
        x, y = party(([X0,bsize],devices[0]))
    
    torch.cuda.empty_cache()

    return x.cpu(),y.cpu()
