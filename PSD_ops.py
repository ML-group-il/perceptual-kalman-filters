import numpy as np

import config
g_th  = config.conf_th
g_eps = config.conf_eps

def lmax(s):
    d,u = np.linalg.eig(s)
    lmax = np.max( np.abs(d) )
    return lmax

def lmin(s):
    d,u = np.linalg.eig(s)
    lmin = np.min( d )
    return lmin
 