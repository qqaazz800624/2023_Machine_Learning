#%%

import os
import glob
import random
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch import optim
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np
import logging
from tqdm import tqdm

from torchvision.utils import save_image
from stylegan2_pytorch import ModelLoader

#%%

n_output = 100
noise  = torch.randn(n_output, 512) 

#%%

ckpts = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
for i in ckpts:
    loader = ModelLoader(base_dir = '/home/u/qqaazz800624/2023_Machine_Learning/HW6',   
                         name = 'default',
                         load_from = i)
    styles  = loader.noise_to_styles(noise, trunc_psi = 0.7)  
    imgs_sample = loader.styles_to_images(styles[:5])
    grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=5)
    #plt.figure()
    plt.imshow(grid_img.permute(1, 2, 0))
    plt.title(f'Sample images generated from step {i}')
    plt.savefig(f'/home/u/qqaazz800624/2023_Machine_Learning/HW6/gradescope/default/images_ckpt_{i}.png', dpi=200)
    plt.show()

#%%



