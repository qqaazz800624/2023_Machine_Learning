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


# seed setting
def same_seeds(seed):
    # Python built-in random module
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(2023)
data_dir = '/neodata/ML/hw6_dataset/'

#%%
# reference: https://github.com/Singyuan/Machine-Learning-NTUEE-2022/blob/master/hw6/hw6.ipynb

# prepare for CrypkoDataset

class CrypkoDataset(Dataset):
    def __init__(self, fnames, transform):
        self.transform = transform
        self.fnames = fnames
        self.num_samples = len(self.fnames)

    def __getitem__(self,idx):
        fname = self.fnames[idx]
        img = torchvision.io.read_image(fname)
        img = self.transform(img)
        return img

    def __len__(self):
        return self.num_samples

def get_dataset(root):
    fnames = glob.glob(os.path.join(root, '*'))
    compose = [
        transforms.ToPILImage(),
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
    ]
    transform = transforms.Compose(compose)
    dataset = CrypkoDataset(fnames, transform)
    return dataset


#%% 
# Show some samples in the dataset
#reference: https://github.com/Singyuan/Machine-Learning-NTUEE-2022/blob/master/hw6/hw6.ipynb

temp_dataset = get_dataset(os.path.join(data_dir, 'faces'))
images = [temp_dataset[i] for i in range(8)]
grid_img = torchvision.utils.make_grid(images, nrow=4)
plt.figure(figsize=(10,10))
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()

#%% 
# Training model in terminal
import os
command_training = 'stylegan2_pytorch --new \
      --name d11948002_hw6_stylegan2 \
      --data /neodata/ML/hw6_dataset/faces \
      --num-train-steps 20000 \
      --results_dir /home/u/qqaazz800624/2023_Machine_Learning/HW6/results/stylegan2 \
      --models_dir /home/u/qqaazz800624/2023_Machine_Learning/HW6/models/stylegan2 \
      --image-size 64 \
      --network-capacity 32'
os.system(command_training)

#%% 
# Interpolation

command_interpolation = 'stylegan2_pytorch--generate-interpolation \
                        -- name d11948002_hw6_stylegan2\
                        -- models_dir /home/u/qqaazz800624/2023_Machine_Learning/HW6/models/stylegan2 \
                        -- interpolation-num-steps 5'
os.system(command_interpolation)

#%%
#reference: https://github.com/Singyuan/Machine-Learning-NTUEE-2022/blob/master/hw6/hw6.ipynb

import torch
from torchvision.utils import save_image
from stylegan2_pytorch import ModelLoader

loader = ModelLoader(
    base_dir = '/home/u/qqaazz800624/2023_Machine_Learning/HW6',   # path to where you invoked the command line tool
    name = 'default'         # the project name, defaults to 'default'
)

n_output = 1000
noise   = torch.randn(n_output, 512) # noise
styles  = loader.noise_to_styles(noise, trunc_psi = 0.7)  # pass through mapping network

imgs_sample = loader.styles_to_images(styles[:10])
grid_img = torchvision.utils.make_grid(imgs_sample.cpu(), nrow=5)
plt.figure()
plt.imshow(grid_img.permute(1, 2, 0))
plt.show()


#%%

img_serial = 0
batch = 10
os.makedirs('images', exist_ok=True)
for i in range(n_output//batch):
    images = loader.styles_to_images(styles[i*batch : (i+1)*batch])
    for img in images:
        save_image(img, f'images/{img_serial+1}.jpg') 
        img_serial += 1

#%%

'''
%cd images
!tar -zcf ../submission.tgz *.jpg
%cd ..
'''

#%%