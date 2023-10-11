##########################
# Perceptual Kalman Filters 
# Harmonic oscillator demo
# by 
# ~
# Implementation of demo from "PKF: online state estimation under a perfect perceptual quality constraint" (2023) by anonymous author(s)

# to execute run: python ./run_osc_demo.py
# or: ./run_demo.sh
########################################

import numpy as np

import torch
import torch.multiprocessing as mp
import multiprocessing.dummy as mpdummy

import pickle
import datetime
from tqdm import tqdm,trange

import matplotlib
matplotlib.use('Qt5Agg', force=True)
import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    'legend.fontsize': 20, 
})
plt.ion()

# my environment
import config
import PSD_ops_torch, PSD_ops
import dynamics, pkf
import perception
from PSD_ops_torch import Tensor


if __name__ == "__main__":
## parameters
    
    args  = lambda:0
    args.workers = 1
    
    T     = 2**8       # Horizon
    bsize = 2**10      # Trials no.

    g_seed = 0

    prec_type = np.float64
    eps = 1e-6
    to = 0              # plot start time
    tsplot = T//32      # 'marker every'
    
    TplotP = T+2 # maximal  vector 
                 # length for Wass. 
                 # distance computation
    TplotDT = 8  # time interval for Wass. 
                 # distance estimation
    
    # experiment identifier (label)
    expid = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
    randid = np.random.randint(99)
    expid += 'r' + str(randid)
    # 
    
    # initiate random states BEFORE
    # initiating multipool
    np.random.seed(   g_seed)
    torch.manual_seed(g_seed)
    sim_seed = np.random.randint(99)

    g_Nworkers = 1  # NOT guaranteed to work 
                    # if > 1
    g_useGPU = True
    g_saveMem = True
    
    mp.set_start_method('spawn', force = True)
    if g_Nworkers > 1:
        multi_pool = mp.Pool(processes=g_Nworkers)
        print(expid,' using multi pool:', g_Nworkers)
    else:
        multi_pool = mpdummy.Pool(processes=g_Nworkers)
        print(expid,' using dummy pool:', g_Nworkers)

    if torch.cuda.is_available() and g_useGPU:
        gpu_n = torch.cuda.device_count()
        devices = [torch.device("cuda:"+str( t%gpu_n ) ) for t in range(8) ]
    else:
        device = torch.device("cpu")
        devices = [torch.device("cpu") for _ in range(8) ]

    # np.random.seed(g_seed)

## harmonic osc.
    omg2 = 2
    dt = 0.5e-2
    A = np.eye(2,dtype = prec_type) + np.array(([0, 1], [-omg2,0])) * dt
    C = np.array(([1.],[0]),dtype = prec_type).T
    C = np.array(([1.],[-1./omg2]),dtype = prec_type).T
    
    ny, nx = C.shape
    R = 2.5e0*np.eye(ny,dtype = prec_type)* dt
    Q = 2.5e0*np.eye(nx,dtype = prec_type)* dt

    P0 = np.eye(nx) * 1e-2

    scale =  1e-0

## preproc.

    qrho  = PSD_ops.lmax(Q)
    
    Q = scale*(np.around(Q/qrho, decimals=3))
    R = scale*(np.around(R/qrho, decimals=3))
    P0 = scale*(np.around(P0/qrho, decimals=3))
    
    Q += np.eye(nx, dtype=prec_type)*PSD_ops.lmax(Q)*1e-3
    R += np.eye(ny, dtype=prec_type)*PSD_ops.lmax(R)*1e-3
    P0 += np.eye(nx, dtype=prec_type)*PSD_ops.lmax(P0)*1e-3
    
    
    A  = PSD_ops_torch.Tensor(A)
    C  = PSD_ops_torch.Tensor(C)
    
    Q  = PSD_ops_torch.Tensor(Q)
    R  = PSD_ops_torch.Tensor(R)
    P0 = PSD_ops_torch.Tensor(P0)
    
## Simulation
    ny, nx = C.size()
    
## reference distribution 

    # np.random.seed(g_seed)
    # torch.manual_seed(g_seed)

    gbTs,Sxx,Mxx = perception.compute_statistics(T,A,Q,P0,TplotP,TplotDT)
    
## groundtruth (reference)
    # np.random.seed(sim_seed)
    # torch.manual_seed(sim_seed)
    
    x,y = dynamics.make_dyn(args,T,A,C,Q,R,P0,bsize,multi_pool = multi_pool)
   
    Gbxref = perception.compute_perception(x,T,Sxx,Mxx,TplotP,TplotDT)
    GBS = np.hstack([Gbxref])
    GBlg = [ r'${x}_{gt}$ (ref.)']
    
    ## Kalman filter
    print('kalmaning...')
    
    xhat, I, SKM, Ps = pkf.kalman(y,A,C,Q,R,P0, output_innovation_cov=True)
    mse_x = dynamics.MSE(x,xhat)
    mse_x_th = np.array( [torch.trace(p).item() for p in Ps] )

    del y
    
    Gbxhat = perception.compute_perception(xhat,T,Sxx,Mxx,TplotP,TplotDT)
    GBS = np.hstack([GBS, Gbxhat])
    GBlg += [ r'$\hat{x}_{kal}^*$']
    
    ## Temporally-incosistent (tic) filter
    print('tic filter...')
    
    # Theoretical error
    lbMSE,_,PXs,PXstars = pkf.lboundMSE(A,Q,P0,SKM,Ps, mp = multi_pool)
    
    Xtic = pkf.tic_kalman(xhat, PXs = PXs , PXstars = PXstars, mp = None)
    mse_Xtic = dynamics.MSE(x,Xtic)

    GbXtic = perception.compute_perception(Xtic,T,Sxx,Mxx,TplotP,TplotDT)
    GBS = np.hstack([GBS, GbXtic])
    GBlg += [ r'$\hat{x}_{tic}$']
    
    print('tic filter...Done!')
        
    if g_saveMem:   
        del  xhat 
        del Xtic
    

## apply filters
    
    
    ## PKF (total cost, AUC)
    print('AUC filter...')

    Pis, _, _ = pkf.pkal_matrices(T,A,Q,P0,SKM, multi_pool = multi_pool)
    Xhat = pkf.pkal_par([I,A,Q,P0,SKM, Pis])
    mse_X = dynamics.MSE(x,Xhat)
    mse_X_th = pkf.thMSE(A,Q,P0,Pis,SKM,Ps)

    GbXhat = perception.compute_perception(Xhat,T,Sxx,Mxx,TplotP,TplotDT)
    GBS = np.hstack([GBS, GbXhat])
    GBlg += [ r'$\hat{x}_{auc}$']
    
    print('AUC filter...Done!')

    ## PKF (terminal cost)
    print('Term. filter...')
    weights =-1
    Pis_T, _, _ = pkf.pkal_matrices(T,A,Q,P0,SKM,weights=weights, multi_pool = multi_pool)
    Xhat_minT = pkf.pkal_par([I,A,Q,P0,SKM,Pis_T])
    mse_X_minT = dynamics.MSE(x,Xhat_minT)
    mse_T_th = pkf.thMSE(A,Q,P0,Pis_T,SKM,Ps)

    GbminT = perception.compute_perception(Xhat_minT,T,Sxx,Mxx,TplotP,TplotDT)
    GBS = np.hstack([GBS, GbminT])
    GBlg += [ r'$\hat{x}_{minT}$']
    
    print('Term. filter....Done!')
 
    ## Optimized coefficients (opt)
    
    with open('./models/hosc_T256_.pkf', 'rb') as handle:
    
        opt = pickle.load(handle)
        pis  = opt['Pis' ]
        phis = opt['Phis']
        
    Xopt, UPSopt, mse_opt_th = pkf.optpkf(I, T,A,Q,P0,SKM, pis,phis)
        
    mse_Xopt = dynamics.MSE(x,Xopt)
    mse_opt_th = np.array(mse_opt_th) + np.array(mse_x_th)
    mse_opt_th_ = mse_opt_th
    
    GbXopt = perception.compute_perception(Xopt,T,Sxx,Mxx,TplotP,TplotDT)
    GBS = np.hstack([GBS, GbXopt])
    GBlg += [ r'$\hat{x}_{opt}$']
    
## errata
    
    mse_x_th = np.array( [torch.trace(p).item() for p in Ps] )
    
## plot
    To = range(to,T)

    fig, ax = plt.subplots(2, squeeze = False)
    plt.suptitle( r'$N_{s} = $'+str(bsize))

    ax[0][0].set_ylabel('MSE (empirical)',  fontsize =28)
    ax[1][0].set_ylabel('MSE (analytical)',  fontsize =28)
    ax[0][0].tick_params(axis='both', which='major', labelsize=18)
    ax[1][0].tick_params(axis='both', which='major', labelsize=18)

    ax[0][0].set_xlabel(r'$k \left( \mathrm{time}\right)$', fontsize = 28)
    ax[1][0].set_xlabel(r'$k \left( \mathrm{time}\right)$', fontsize = 28)
    
    
    ax[0][0].plot(To, mse_x[to:],"b:", label=r'$\hat{x}_{\mathrm{kal}}^*$', lw=3)
    ax[0][0].plot(To, mse_Xtic[to:],"g:", label=r'$\hat{x}_{\mathrm{tic}}$', lw=3)
    ax[0][0].plot(To, mse_X[to:],'g', label=r'$\hat{x}_{\mathrm{auc}}$', lw=3)
    
    ax[0][0].plot(To, mse_X_minT[to:], linestyle='-', marker='*', color='b', label=r'$\hat{x}_{\mathrm{minT}}$', markevery=tsplot, markersize=16, lw=3)
    ax[0][0].scatter(T-1, mse_X_minT[T-1],marker="*", s=300, color='b')

    ax[0][0].plot(To, mse_Xopt[to:],'r', label=r'$\hat{x}_{\mathrm{opt}}$', lw=3)
    ax[0][0].scatter(T-1, mse_Xopt[T-1],marker="o", s=100, c='r')
    
    
    ax[1][0].plot(To, mse_x_th[to:],"b:", label=r'$\hat{x}_{\mathrm{kal}}^*$', lw=3)
    ax[1][0].plot(To, lbMSE[to:],"g:", label=r'$\hat{x}_{\mathrm{tic}}$', lw=3)
    ax[1][0].plot(To, mse_X_th[to:],'g', label=r'$\hat{x}_{\mathrm{auc}}$', lw=3)

    ax[1][0].plot(To, mse_T_th[to:], linestyle='-', marker='*', color='b', label=r'$\hat{x}_{\mathrm{minT}}$', markevery=tsplot, markersize=16, lw=3)
    
    ax[1][0].scatter(T-1, mse_T_th[T-1],marker="*", s=300, color='b')

    ax[1][0].plot(To, mse_opt_th_[to:],'b', lw=3)
    
    ax[1][0].plot(To, mse_opt_th[to:],'r', label=r'$\hat{x}_{\mathrm{opt}}$', lw=3)
    ax[1][0].scatter(T-1, mse_opt_th[T-1],marker="o", s=100, c='r')
    ax[1][0].axhline(mse_opt_th[-1],xmin=0, ls = "-.", label=r'$\hat{x}_{\mathrm{opt}}$', lw=2.5)
    
    
    ax[0][0].legend()
    ax[1][0].legend()

    plt.legend()

## close-ups plot
    fig, ax = plt.subplots(1, squeeze = False)
    plt.suptitle( r'Ns = '+str(bsize))
    
    ax[0][0].set_ylabel('MSE (analytical)',  fontsize =40)
    ax[0][0].tick_params(axis='both', which='major', labelsize=28)
    ax[0][0].set_xlabel(r'$k \left( \mathrm{time}\right)$', fontsize = 40)
    
    
    tocu = 0#T//2
    Tocu = range(tocu,T)
    
    ax[0][0].plot(Tocu, mse_opt_th_[tocu:],'b', lw=3)
    
    ax[0][0].plot(Tocu, mse_opt_th[tocu:],'r', label=r'$\hat{x}_{\mathrm{opt}}$', lw=3)
    ax[0][0].scatter(T-1, mse_opt_th[T-1],marker="o", s=100, c='r')
    ax[0][0].axhline(mse_opt_th[-1],xmin=0, ls = "-.", lw=2.5)
    
    ax[0][0].plot(Tocu, mse_x_th[tocu:],"b:", label=r'$\hat{x}_{\mathrm{kal}}^*$', lw=3)
    ax[0][0].plot(Tocu, lbMSE[tocu:],"g:", label=r'$\hat{x}_{\mathrm{tic}}$', lw=3)
    ax[0][0].plot(Tocu, mse_X_th[tocu:],'g', label=r'$\hat{x}_{\mathrm{auc}}$', lw=3)
    
    ax[0][0].plot(Tocu, mse_T_th[tocu:], linestyle='-', marker='*', color='b', label=r'$\hat{x}_{\mathrm{minT}}$', markevery=tsplot, markersize=16, lw=3)
    ax[0][0].scatter(T-1, mse_T_th[T-1],marker="*", s=300, color='b')
    
    # consistency gap
    plt.text( T-T/2, 0.65*lbMSE[-1] +  0.35*mse_opt_th[T-1] , 'Gap due to\n temporal \n consistency',ha="center", va = "center", fontsize = 44,bbox=dict(facecolor='orange', edgecolor='None', pad=.25, alpha=.85)) 
 
    plt.annotate('', xy=(T-1, lbMSE[-1]), xytext=(T-1, mse_opt_th[T-1]) ,
horizontalalignment="center",
            arrowprops=dict(arrowstyle='<->', color='orange', lw=5))

    # PKF gap
    plt.text( T-T/2, 0.65*mse_T_th[-1] +  0.35*mse_opt_th[T-1] , 'Gap due \n to ' + r' $\Phi_k = 0$',ha="center", va = "center", fontsize = 44,bbox=dict(facecolor='orange', edgecolor='None', pad=.25, alpha=.85))
     
    plt.annotate('', xy=(T-1, mse_T_th[-1]), xytext=(T-1, mse_opt_th[T-1]) ,
horizontalalignment="center",
            arrowprops=dict(arrowstyle='<->', color='orange', lw=5))

    ax[0][0].legend(loc='lower left', fontsize=38)
    
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    plt.tight_layout()

 ## plot Wasserstein (Gelbrich) distance
    
    # labels arrangement
    def argb(gb, delta = 0.001, ylim=None):
        gbi = np.argsort(gb)
        delta = gb[gbi[1]] / 0.9
        
        delta = (ylim[1]-ylim[0])/18.
        gb[gbi[0]] -= delta*1.05
        gb[gbi[0]] = np.max([ gb[gbi[0]], ylim[0] ])
         
        for i in range(1,len(gb)):
            if gb[gbi[i]] - gb[gbi[i-1]] < delta:
                gb[gbi[i]] = gb[gbi[i-1]] + delta
        return gb
    
    
    fig, ax = plt.subplots(1, squeeze = False)
    
    TplotP_Nparams = nx*TplotP*(nx*TplotP+1)/2
    GBS = np.hstack([gbTs,GBS])
    
    gb2 = np.vstack(GBS)
    plt.xticks(gb2[::2,0])
    plt.plot(gb2[:,0],np.sqrt(gb2[:,1:]), marker="D")
    
    gblax_ = np.sqrt(gb2[-1,1:])
    gblaxm = np.mean(gblax_)
    lg = GBlg[0:]

    gblax = argb(np.copy(gblax_), ylim = ax[0][0].get_ylim())
            
    ax[0][0].set_xlim([0,1.05*T])
    ax2 = ax[0][0].twinx()
    ax2.set_ylim( ax[0][0].get_ylim() )
    ax2.set_xlim( ax[0][0].get_xlim() )

    plt.xlim( ax[0][0].get_xlim() )
    plt.ylim( ax[0][0].get_ylim() )
    plt.yticks(gblax, lg)
    
    ax[0][0].tick_params(axis='x', which='major', labelsize=18)
    ax2.tick_params(axis='y', which='major', labelsize=25)
    
    #ylabel (left)
    s_TplotPs = 'k-'+str(TplotP-1) if TplotP > 1 else 'k'
    s_TplotPs = s_TplotPs if TplotP < T else '0'
    s_topk,= 'k' if TplotP > 1 else ' '
    s_ylabel_left = 'Emp. Wasserstein distance ' + r'$\frac{\hat{d}_P\left( {X}^{%s}_{%s},\hat{X}^{%s}_{%s}\right)}{\sqrt{k+1}}$' % (s_topk, s_TplotPs,s_topk, s_TplotPs)
    
    ax[0][0].set_ylabel(s_ylabel_left, fontsize = 26)
    ax[0][0].set_xlabel(r'$k \left( \mathrm{time}\right)$', fontsize = 26)
    
    plt.suptitle('vec.len ' + str(TplotP) + ', nparams ' + str(TplotP_Nparams)+", nsamples " + str(bsize))
    
    # arrows to right labels
    for p in range(len(gblax)):
        xa = ax[0][0].get_xlim()[-1]
        ya =  gblax[p]
        dx =  gb2[-1,0] - xa
        dy =  gblax_[p] - ya
        ylim = ax[0][0].get_ylim()
        delta = ylim[1]-ylim[0]
        lwy = delta*5e-2
        ax[0][0].arrow(xa,ya,dx,dy, ls=(0,(1,5)),lw=lwy, alpha = 0.85)
    

## epilogue

   
    plt.ioff()
    plt.show()

    multi_pool.close()
    multi_pool.join()