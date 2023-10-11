
import numpy as np
import torch
import argparse
import sys

conf_th = 1e-9
conf_th = 1e-8
conf_th = 5e-10
conf_eps = 7.5e-8 #1e-6#5e-5

conf_prec_type       =  np.float64 #
conf_prec_type_torch = torch.float64