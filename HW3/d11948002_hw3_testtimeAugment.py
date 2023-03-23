#%%

# Import necessary packages.
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
# "ConcatDataset" and "Subset" are possibly useful when doing semi-supervised learning.
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
# This is for the progress bar.
from tqdm.auto import tqdm
import random
from randaugment import ImageNetPolicy


myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

test_tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.490, 0.455, 0.405], [0.230, 0.225, 0.225])
                ])

test_tfm1 = transforms.Compose([
                transforms.RandomRotation(10), 
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
                transforms.RandomHorizontalFlip(p=0.5), 
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.490, 0.455, 0.405], [0.230, 0.225, 0.225])
                ])

test_tfm2 = transforms.Compose([
                transforms.RandomRotation(20), 
                transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
                transforms.RandomHorizontalFlip(p=0.2),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.490, 0.455, 0.405], [0.230, 0.225, 0.225]) 
                ])

# test_tfm3 = transforms.Compose([
#                 transforms.RandomRotation(20), 
#                 transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
#                 transforms.RandomHorizontalFlip(p=0.3),
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.490, 0.455, 0.405], [0.230, 0.225, 0.225]) 
#                 ])

# test_tfm4 = transforms.Compose([
#                 transforms.RandomRotation(5), 
#                 transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.1),
#                 transforms.RandomHorizontalFlip(p=0.3),
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.490, 0.455, 0.405], [0.230, 0.225, 0.225]) 
#                 ])


# test_tfm5 = transforms.Compose([
#                 transforms.RandomRotation(15), 
#                 transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
#                 transforms.RandomHorizontalFlip(p=0.2),
#                 transforms.Resize((224, 224)),
#                 transforms.ToTensor(),
#                 transforms.Normalize([0.490, 0.455, 0.405], [0.230, 0.225, 0.225]) 
#                 ])


class FoodDataset(Dataset):

    def __init__(self,path,tfm=test_tfm,files = None):
        super(FoodDataset).__init__()
        self.path = path
        self.files = sorted([os.path.join(path,x) for x in os.listdir(path) if x.endswith(".jpg")])
        if files != None:
            self.files = files
            
        self.transform = tfm
  
    def __len__(self):
        return len(self.files)
  
    def __getitem__(self,idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transform(im)
        
        try:
            label = int(fname.split("/")[-1].split("_")[0])
        except:
            label = -1 # test has no label
            
        return im,label

# "cuda" only when GPUs are available.
device = "cuda:1" if torch.cuda.is_available() else "cuda:0"


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = models.resnet101(weights=False).to(device)
        self.fc = nn.Sequential(
                        nn.Linear(1000, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 11)
                        ).to(device)

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


model_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW3/ckpts/model4_best.ckpt'

# The number of batch size.
batch_size = 64

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = FoodDataset("/neodata/ML/hw3_dataset/test", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

test_set1 = FoodDataset("/neodata/ML/hw3_dataset/test", tfm=test_tfm1)
test_loader1 = DataLoader(test_set1, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
test_set2 = FoodDataset("/neodata/ML/hw3_dataset/test", tfm=test_tfm2)
test_loader2 = DataLoader(test_set2, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
# test_set3 = FoodDataset("/neodata/ML/hw3_dataset/test", tfm=test_tfm3)
# test_loader3 = DataLoader(test_set3, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
# test_set4 = FoodDataset("/neodata/ML/hw3_dataset/test", tfm=test_tfm4)
# test_loader4 = DataLoader(test_set4, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
# test_set5 = FoodDataset("/neodata/ML/hw3_dataset/test", tfm=test_tfm5)
# test_loader5 = DataLoader(test_set5, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
#test_loaders = [test_loader, test_loader1, test_loader2, test_loader3, test_loader4, test_loader5]
test_loaders = [test_loader, test_loader1, test_loader2]

model = MyModel().to(device)
model.load_state_dict(torch.load(model_path))


predict = []
logits_list = []

for loader in test_loaders:
    logit_temp = []
    for batch in tqdm(test_loader):
        imgs, labels = batch

        with torch.no_grad():
            inputs = imgs.to(device)
            logits = model(inputs)
        logit_temp.extend(logits)

    logits_list.append(logit_temp)
#%%

tensor_0 = torch.stack(logits_list[0])
tensor_1 = torch.stack(logits_list[1])
tensor_2 = torch.stack(logits_list[2])
# tensor_3 = torch.stack(logits_list[3])
# tensor_4 = torch.stack(logits_list[4])
# tensor_5 = torch.stack(logits_list[5])

logits = (tensor_0*0.8 + tensor_1[1]*0.1 + tensor_2[2]*0.1)

# Take the class with greatest logit as prediction and record it.
predict.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

#%%

with open('/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_tta.csv', 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(predict):
        f.write(f"{i},{y}\n")


#%%
