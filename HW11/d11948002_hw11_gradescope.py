#%%

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
 
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
 
from rich.progress import Progress, TextColumn, BarColumn, TimeElapsedColumn, TimeRemainingColumn, track
import time

_exp = 'gradescope'
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cuda:0')


def same_seeds(seed):
    # random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(36)


source_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Lambda(lambda x: cv2.Canny(np.array(x), 170, 300)),
    transforms.ToPILImage(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])

target_transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((32, 32)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15, fill=(0,)),
    transforms.ToTensor(),
])
 
source_dataset = ImageFolder('/neodata/ML/hw11_dataset/train_data', transform=source_transform)
target_dataset = ImageFolder('/neodata/ML/hw11_dataset/test_data', transform=target_transform)
 
source_dataloader = DataLoader(source_dataset, batch_size=32, shuffle=True)
target_dataloader = DataLoader(target_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(target_dataset, batch_size=128, shuffle=False)



class FeatureExtractor(nn.Module):

    def __init__(self):
        super(FeatureExtractor, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 512, 3, 1, 1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        
    def forward(self, x):
        x = self.conv(x).squeeze()
        return x

class LabelPredictor(nn.Module):

    def __init__(self):
        super(LabelPredictor, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.ReLU(),

            nn.Linear(512, 10),
        )

    def forward(self, h):
        c = self.layer(h)
        return c

class DomainClassifier(nn.Module):

    def __init__(self):
        super(DomainClassifier, self).__init__()

        self.layer = nn.Sequential(
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),

            nn.Linear(512, 1),
        )

    def forward(self, h):
        y = self.layer(h)
        return y


#%% Gradescope

import numpy as np
import matplotlib.pyplot as plt
from sklearn import manifold

source_data, source_label = next(iter(source_dataloader))
# source_data, source_label = next(iter(target_dataloader))
source_data.shape

#%%

# Q1.	Visualize distribution of features across different classes.

#step 1

feature_extractor = FeatureExtractor().to(device)
#feature_extractor.load_state_dict(torch.load("models/extractor_model1.bin", map_location=device))
feature_extractor.load_state_dict(torch.load("models/extractor_model1_early.bin", map_location=device))
#feature_extractor.load_state_dict(torch.load("models/extractor_model1_mid.bin", map_location=device))

X_source = np.empty((0,512), float) 
Y_source = np.empty((0), int) 
for i, (source_data, source_label) in enumerate(source_dataloader):
    feat = feature_extractor(source_data.to(device))
    feat = feat.cpu().detach().numpy()
    X_source = np.append(X_source, feat, axis=0)
    Y_source = np.append(Y_source, source_label.detach().numpy())

#step2

# process extracted features with t-SNE
X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X_source)

# Normalization the processed features 
x_min, x_max = X_tsne.min(0), X_tsne.max(0)
X_norm = (X_tsne - x_min) / (x_max - x_min)

#step3

# Data Visualization
# Use matplotlib to plot the distribution
# The shape of X_norm is (N,2)
plt.figure(figsize=(8,6))
colormap = np.array(plt.cm.tab10.colors)
plt.scatter(X_norm[:, 0], X_norm[:, 1], s=3, c=colormap[Y_source])
# plt.savefig('class-fianl.png')


#%%

# Q2.	Visualize distribution of features across different domains

feature_extractor = FeatureExtractor().to(device)
feature_extractor.load_state_dict(torch.load("models/extractor_model1.bin", map_location=device))
#feature_extractor.load_state_dict(torch.load("models/extractor_model1_early.bin", map_location=device))
#feature_extractor.load_state_dict(torch.load("models/extractor_model1_mid.bin", map_location=device))


X_target = np.empty((0,512), int) # initial
target_iter = iter(target_dataloader)
cnt = 0
while cnt < 2.0*len(Y_source):
    target_data, target_label = next(target_iter)
    feat = feature_extractor(target_data.to(device))
    feat = feat.cpu().detach().numpy()
    X_target = np.append(X_target, feat, axis=0)
    cnt += len(target_data)

X_all = np.append(X_source, X_target, axis=0)
cmap_source = [0] * len(X_source)
cmap_target = [1] * len(X_target)
cmap_all = cmap_source+cmap_target
cmap_all = np.array(cmap_all)

# process extracted features with t-SNE
Xall_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(X_all)

# Normalization the processed features 
xall_min, xall_max = Xall_tsne.min(0), Xall_tsne.max(0)
Xall_norm = (Xall_tsne - xall_min) / (xall_max - xall_min)

# Data Visualization
# Use matplotlib to plot the distribution
# The shape of X_norm is (N,2)
plt.figure(figsize=(8,6))
plt.scatter(Xall_norm[cmap_all==0, 0], Xall_norm[cmap_all==0, 1], s=1, c='r')
plt.scatter(Xall_norm[cmap_all==1, 0], Xall_norm[cmap_all==1, 1], s=1, c='b')
plt.legend(['source', 'target'])
# plt.savefig('source-final.png')

#%%


