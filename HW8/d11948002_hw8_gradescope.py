#%%

import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam, AdamW
from qqdm import qqdm, format_str
import pandas as pd
from torchsummary import summary

#%%

_exp = 'model2'
train = np.load('/neodata/ML/ml2023spring-hw8/trainingset.npy', allow_pickle=True)
test = np.load('/neodata/ML/ml2023spring-hw8/testingset.npy', allow_pickle=True)

print(train.shape)
print(test.shape)
#%%
device = "cuda:0" if torch.cuda.is_available() else "cuda:3"

def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

same_seeds(48763)


#%%

#references: https://github.com/Singyuan/Machine-Learning-NTUEE-2022/blob/master/hw8/hw8.ipynb
class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.LeakyReLU(0.1), 
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1), 
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 128), 
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
        )

        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 64 * 64 * 3), 
            nn.Tanh()
        )

    def forward(self, x):
        code = self.encoder(x)
        # Adjust Latent Repr for Report Image
        # code = target_code
        y = self.decoder(code)
        # return code, y
        return y

#references: https://github.com/pai4451/ML2021/blob/main/hw8/HW08.ipynb

class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),         
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),        
            nn.ReLU(),
			nn.Conv2d(24, 48, 4, stride=2, padding=1),         
            nn.ReLU(),
        )   # Hint:  dimension of latent space can be adjusted
        self.decoder = nn.Sequential(
			nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.ReLU(),
			nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),    
            nn.ReLU(),
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        # Hint: can add more layers to encoder and decoder
        self.decoder = nn.Sequential(
			      nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), 
                  nn.ReLU(),
			      nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), 
                  nn.ReLU(),
                  nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1), 
                  nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD

#%%

class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        
        self.transform = transforms.Compose([
          transforms.Lambda(lambda x: x.to(torch.float32)),
          transforms.Lambda(lambda x: 2. * x/255. - 1.),
        ])
        
    def __getitem__(self, index):
        x = self.tensors[index]
        
        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)



#%%

# Training hyperparameters
num_epochs = 200
batch_size = 256 # Hint: batch size may be lower
learning_rate = 5e-4

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'fcn'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model_classes = {'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'vae': VAE()}
model = model_classes[model_type].to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from torch.optim.lr_scheduler import StepLR

total_steps = len(train_dataloader) * num_epochs
break_steps = int(0.05 * total_steps)
scheduler = StepLR(optimizer, step_size=break_steps, gamma=0.95)

#%%

print(model)


#%%

from torchsummary import summary
summary(model, (64, 12288))


#%%
eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')


#%%

# load trained model
checkpoint_path = f'models/last_model_{_exp}.pt'
model = torch.load(checkpoint_path)
model.eval()

#%%

import matplotlib.pyplot as plt
import torchvision
train_dataloader_fix = DataLoader(train_dataset, batch_size=batch_size)
data = next(iter(train_dataloader_fix))
img = data.float().to(device)
grid_img = torchvision.utils.make_grid(0.5*img.cpu()[5:9]+0.5, nrow=2)
plt.figure(figsize=(5,5))
plt.imshow(grid_img.permute(1, 2, 0))
plt.savefig("images/original.png")
plt.show()


#%%

imgflat = img.view(img.shape[0], -1)
imgflat2 = model.encoder(imgflat)
outputflat = model.decoder(imgflat2)
output = outputflat.view(img.shape[0], 3, 64, 64)

grid_img = torchvision.utils.make_grid(0.5*output.cpu()[5:9]+0.5, nrow=2)
plt.figure(figsize=(5,5))
plt.imshow(grid_img.permute(1, 2, 0))
plt.savefig("images/rebuilt.png")
plt.show()

#%%
imgflat = img.view(img.shape[0], -1)
imgflat2 = model.encoder(imgflat)
imgflat2[:, 2] = imgflat2[:, 2]*2
outputflat = model.decoder(imgflat2)
output = outputflat.view(img.shape[0], 3, 64, 64)

grid_img = torchvision.utils.make_grid(0.5*output.cpu()[5:9]+0.5, nrow=2)
plt.figure(figsize=(5,5))
plt.imshow(grid_img.permute(1, 2, 0))
plt.savefig("images/rebuilt-0.png")
plt.show()

#%%
imgflat = img.view(img.shape[0], -1)
imgflat2 = model.encoder(imgflat)
imgflat2[:, 3] = imgflat2[:, 3]*3
outputflat = model.decoder(imgflat2)
output = outputflat.view(img.shape[0], 3, 64, 64)

grid_img = torchvision.utils.make_grid(0.5*output.cpu()[5:9]+0.5, nrow=2)
plt.figure(figsize=(5,5))
plt.imshow(grid_img.permute(1, 2, 0))
plt.savefig("images/rebuilt-1.png")
plt.show()



#%%




#%%




#%%





#%%





#%%