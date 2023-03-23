#%%

_exp_name = "sample"
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


#%%

myseed = 6666  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)


#%%

# Normally, We don't need augmentations in testing and validation.
# All we need here is to resize the PIL image and transform it into Tensor.
#reference link:
#https://github.com/Joshuaoneheart/ML2021-HWs
#https://github.com/pai4451/ML2021
test_tfm = transforms.Compose([
                transforms.Resize((224, 224)),
                # transforms.Resize((255, 255)),
                # transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.490, 0.455, 0.405], [0.230, 0.225, 0.225]) #把 [channel, height, width] 的 mean及std 標準化
                ])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
#reference link:
#https://github.com/Joshuaoneheart/ML2021-HWs
#https://github.com/pai4451/ML2021
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.RandomRotation(30), #對圖片從 (-30,30)之間隨機選擇旋轉角度
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2, scale=(0.8, 0.8)), 
    
    # RandomAffine: 對圖片進行仿射變
    # degree:不對中心旋轉
    # translate:對水平與垂直平移(斜移)0.2
    # shear: 把圖片弄得有點像平行四邊形看是對x軸或是y軸,先在x軸,在 (-0.2, 0.2)之間隨機選擇錯切角度
    # scale: 把寬與高的圖片的比例都縮小成原本的 0.8
    
    transforms.RandomHorizontalFlip(p=0.5), #將一半的圖片進行水平方向翻轉，因為訓練的圖片主要是食物，水平翻轉應該也要能認得出來是什麼食物
    # transforms.RandomResizedCrop((224, 224)),
    # transforms.RandomHorizontalFlip(),
    # ImageNetPolicy(),
    transforms.Resize((224, 224)),
    # You may add some transforms here.
    #transforms.RandomHorizontalFlip(p=0.5), #將一半的圖片進行水平方向翻轉，因為訓練的圖片主要是食物，水平翻轉應該也要能認得出來是什麼食物
    #transforms.RandomCrop((128,128), padding = 10),
    # transforms.RandomAffine(degrees=(-20, 20),translate=(0.1, 0.3),scale=(0.5, 0.75)),
    # transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0,hue=0),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize([0.490, 0.455, 0.405], [0.230, 0.225, 0.225]) #把 [channel, height, width] 的 mean及std 標準化
])


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
device = "cuda:2" if torch.cuda.is_available() else "cuda:0"

#%%

class MyModel1(nn.Module):
    def __init__(self):
        super(MyModel1, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = models.densenet121(weights=False).to(device)
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


class MyModel2(nn.Module):
    def __init__(self):
        super(MyModel2, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = models.resnet50(weights=False).to(device)
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


class MyModel3(nn.Module):
    def __init__(self):
        super(MyModel3, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = models.resnet34(weights=False).to(device)
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

class MyModel4(nn.Module):
    def __init__(self):
        super(MyModel4, self).__init__()
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
    

class MyModel5(nn.Module):
    def __init__(self):
        super(MyModel5, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = models.resnet18(weights=False).to(device)
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

class MyModel6(nn.Module):
    def __init__(self):
        super(MyModel6, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = models.resnet18(weights=False).to(device)
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


class MyModel7(nn.Module):
    def __init__(self):
        super(MyModel7, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = models.vgg11(weights=False).to(device)
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
    
class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]
            
            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(25088, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 11)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)
#%%

model1_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW3/ckpts/model1_best.ckpt'
model2_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW3/ckpts/model2_best.ckpt'
model3_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW3/ckpts/model3_best.ckpt'
model4_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW3/ckpts/model4_best.ckpt'
model5_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW3/ckpts/model5_best.ckpt'
model6_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW3/ckpts/model6_best.ckpt'
model7_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW3/ckpts/model7_best.ckpt'
model8_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW3/ckpts/gradescope_hw3_best.ckpt'

# The number of batch size.
batch_size = 64

#%%

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = FoodDataset("/neodata/ML/hw3_dataset/test", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


#%%

model1 = MyModel1().to(device)
model1.load_state_dict(torch.load(model1_path))
model2 = MyModel2().to(device)
model2.load_state_dict(torch.load(model2_path))
model3 = MyModel3().to(device)
model3.load_state_dict(torch.load(model3_path))
model4 = MyModel4().to(device)
model4.load_state_dict(torch.load(model4_path))
model5 = MyModel5().to(device)
model5.load_state_dict(torch.load(model5_path))
model6 = MyModel6().to(device)
model6.load_state_dict(torch.load(model6_path))
model7 = MyModel7().to(device)
model7.load_state_dict(torch.load(model7_path))
model8 = Classifier().to(device)
model8.load_state_dict(torch.load(model8_path))


#%%
#set models to evaluation mode
predict = []
model1.eval() 
model2.eval() 
model3.eval() 
model4.eval() 
model5.eval() 
model6.eval() 
model7.eval() 
model8.eval() 

#reference: https://github.com/pai4451/ML2021/blob/main/hw3/Ensemble2.ipynb

for batch in tqdm(test_loader):
    imgs, labels = batch

    with torch.no_grad():
        inputs = imgs.to(device)
        logits1 = model1(inputs)
        logits2 = model2(inputs)
        logits3 = model3(inputs)
        logits4 = model4(inputs)
        logits5 = model5(inputs)
        logits6 = model6(inputs)
        logits7 = model7(inputs)
        logits8 = model8(inputs)
        #logits = (logits1 + logits2 + logits3 + logits4 + logits5 + logits6 ) / 6
        logits = (logits1 + logits2 + logits3 + logits4 + logits5 + logits6 + logits7 + logits8) / 8

    # Take the class with greatest logit as prediction and record it.
    predict.extend(logits.argmax(dim=-1).cpu().numpy().tolist())

#%%

with open('/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_ensemble2.csv', 'w') as f:
    f.write('Id,Category\n')
    for i, y in enumerate(predict):
        f.write(f"{i},{y}\n")


#%%
