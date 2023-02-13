# import libs
from __future__ import print_function
import matplotlib.pyplot as plt
#% matplotlib inline

import os
import cv2
#os.environ['CUDA_VISIBLE_DEVICES'] = '3'

import numpy as np
from models import *
from scipy import io

import torch
import torch.optim
from PIL import Image
from utils.denoising_utils import *
from models.Unet2D_simulation import UNet

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
dtype = torch.cuda.FloatTensor

imsize = -1
PLOT = True
sigma = 25
sigma_ = sigma/255.
s1 = 104
s2 = 104
s3 = 24

# load the \alpha image as the label
fp = open(r'noise_input_2D.img','rb')
img_noisy_np = np.fromfile(fp,dtype=np.float32).reshape((s1, s2))
img_max = img_noisy_np.max()
img_noisy_np = img_noisy_np / img_max
img_noisy_np = torch.from_numpy(img_noisy_np)
img_noisy_np = img_noisy_np.unsqueeze(0)
img_noisy_np = img_noisy_np.numpy()

INPUT = 'noise'  # 'meshgrid'
pad = 'reflection'
OPT_OVER = 'net'  # 'net,input'

# load sub-iteration number
matr = io.loadmat(r'trained\sub_iteration_number_frame.mat')
Dip_iter = matr['sub_iteration_number_frame']
Dip_iter = Dip_iter.max()
# Dip_iter = 201
reg_noise_std = 1. / 30.  # set to 1./20. for sigma=50
LR = 1e-3
OPTIMIZER = 'adam'  # 'LBFGS'
#OPTIMIZER = 'LBFGS'
show_every = Dip_iter-1
exp_weight = 0.99
num_iter = Dip_iter
input_depth = 3
figsize = 5
in_channel = 3
inter_channel = 16
out_channel = 1
net = UNet(in_channel, inter_channel, out_channel)
net = net.type(dtype)

# load sub-iteration number and pass it to choose which model
matr1 = io.loadmat(r'trained\sub_iteration_frame.mat')
iter = matr1['sub_iteration_frame']
iter = iter.max()

# the first sub-iteration in the first epoch of Recon. is randomly selected if iter = 0 or can use better initialization.
if iter != 0:
    f1 = os.path.join(r'trained\OT_model', 'DIP_Unet_{}iter.ckpt'.format(iter-1))
    net.load_state_dict(torch.load(f1))


# load the weight image (fixed)
fp2 = open(r'weight_img_2D.img','rb')
weight_img = np.fromfile(fp2,dtype=np.float32).reshape((s1, s2))
weight_img_max = weight_img.max()
weight_img = weight_img / weight_img_max
weight_img = torch.from_numpy(weight_img)
weight_img = weight_img.unsqueeze(0)
weight_img = weight_img.numpy()


# load CIP image or random noise
#net_input = get_noise(1, INPUT, (img_pil.size[1], img_pil.size[0])).type(dtype).detach()
fp1 = open(r'Prior.img','rb')
img_noisy_np1 = np.fromfile(fp1, dtype=np.float32).reshape((in_channel,s1,s2))
img_noisy_np1 = img_noisy_np1 / img_noisy_np1.max()
img_noisy_np1 = torch.from_numpy(img_noisy_np1)
img_noisy_np1 = img_noisy_np1.numpy()
net_input = np_to_torch(img_noisy_np1).type(dtype)


# Compute number of parameters
s = sum([np.prod(list(p.size())) for p in net.parameters()])
#print('Number of params: %d' % s)

# Loss
#mse = torch.nn.MSELoss().type(dtype)
Poisson_loss = torch.nn.PoissonNLLLoss(log_input=False,reduction='none').type(dtype)
img_noisy_torch = np_to_torch(img_noisy_np).type(dtype)
weight_img_torch = np_to_torch(weight_img).type(dtype)
net_input_saved = net_input.detach().clone()
noise = net_input.detach().clone()
out_avg = None
last_net = None
psrn_noisy_last = 0

i = 0

loss = []
def closure():

    global i, out_avg, psrn_noisy_last, last_net, net_input
    out = net(net_input)
    # Smoothing
    if out_avg is None:
        out_avg = out.detach()
    else:
        out_avg = out_avg * exp_weight + out.detach() * (1 - exp_weight)

    total_loss = torch.mean(weight_img_torch * Poisson_loss(out, img_noisy_torch))
    net.zero_grad()
    total_loss.backward()
    loss.append(total_loss.item())

    if PLOT and i % show_every == 0:
        f = os.path.join(r'trained\OT_model', 'DIP_Unet_{}iter.ckpt'.format(i))
        torch.save(net.state_dict(), f)
        #print('Iteration %05d    Loss %f ' % (i, total_loss.item()))

    i += 1
    return total_loss

p = get_params(OPT_OVER, net, net_input)
optimize(OPTIMIZER, p, closure, LR, num_iter)
out_np = torch_to_np(net(net_input))
denor_out = out_np * img_max
fp3 = open('DIP_output_2D.img','wb')
denor_out.tofile(fp3)