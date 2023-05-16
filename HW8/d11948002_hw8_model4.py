#%%

import random
import numpy as np
import torch
from torch import nn, rand
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F

from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam, AdamW
from qqdm import qqdm, format_str
import pandas as pd
from torchsummary import summary


_exp = 'model4'
train = np.load('/neodata/ML/ml2023spring-hw8/trainingset.npy', allow_pickle=True)
test = np.load('/neodata/ML/ml2023spring-hw8/testingset.npy', allow_pickle=True)

print(train.shape)
print(test.shape)

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
            #nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(256),
            nn.Linear(256, 64), 
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            #nn.BatchNorm1d(1024),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Tanh()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


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


class MultiEncoderAutoencoder(nn.Module):
    def __init__(self):
        super(MultiEncoderAutoencoder, self).__init__()
        
        # First Encoder
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 64)
        )
        
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(64, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 64 * 64 * 3), 
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        encoded = self.encoder(x)
        reconstruct = self.decoder(encoded)
        pred = self.classifier(reconstruct)
        
        return pred, encoded, reconstruct



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
num_epochs = 125
batch_size = 128 # Hint: batch size may be lower
learning_rate = 1e-3

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)


# Model
model_type = 'multi'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model_classes = {'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'multi': MultiEncoderAutoencoder()}
model = model_classes[model_type].to(device)

# Loss and optimizer
reconstruction_criterion = nn.MSELoss()
classification_criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts

#total_steps = len(train_dataloader) * num_epochs
scheduler = StepLR(optimizer, step_size=25, gamma=0.95)

#%%

best_loss = np.inf
model.train()
from tqdm import tqdm

# qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
# for epoch in qqdm_train:
for epoch in range(num_epochs):
    tot_loss = list()

    # for data in train_dataloader:
    train_pbar = tqdm(train_dataloader, position=0, leave=True)
    for data in train_pbar:

        # ===================loading=====================
        img = data.float().to(device)
        img = img.view(img.shape[0], -1)

        # ===================forward=====================
        #noisy_inputs = img + torch.randn_like(img) * 0.1  # Add random noise

        real_output, latent_vector, reconstruct = model(img)
        reconstruct = reconstruct.view(reconstruct.shape[0], -1)

        real_label = torch.zeros_like(real_output).to(device)  # label 0: real images
        fake_output = model.decoder(torch.rand_like(latent_vector).to(device))
        fake_output = model.classifier(fake_output)
        fake_label = torch.ones_like(fake_output).to(device)  # label 1: fake images

        classifier_output = torch.cat((real_output, fake_output))
        classifier_label = torch.cat((real_label, fake_label))

        reconstruction_loss = reconstruction_criterion(reconstruct, img)
        classification_loss = classification_criterion(classifier_output, classifier_label)
        loss = reconstruction_loss + classification_loss


        tot_loss.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================step log====================
        train_pbar.set_description(f'[{epoch+1}/{num_epochs}]')
        train_pbar.set_postfix({'loss': loss.detach().item()})
    
    # ===================adjust lr====================
    scheduler.step()
    
    # ===================save_best====================
    mean_loss = np.mean(tot_loss)
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(model, f'models/best_model_{_exp}.pt')
        print(f'[{epoch+1}/{num_epochs}]: Train loss: {mean_loss:.5f}, Reconstruction loss: {reconstruction_loss:.5f}, classification loss: {classification_loss:.5f}, lr = {scheduler.get_last_lr()[0]:.5f} <-- Best model')
    else:
        print(f'[{epoch+1}/{num_epochs}]: Train loss: {mean_loss:.5f}, Reconstruction loss: {reconstruction_loss:.5f}, classification loss: {classification_loss:.5f}, lr = {scheduler.get_last_lr()[0]:.5f}')



#%%

eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
reconstruction_loss = nn.MSELoss(reduction='none')
#classification_loss = nn.CrossEntropyLoss(reduction='none')

# load trained model
checkpoint_path = f'models/best_model_{_exp}.pt'
model = torch.load(checkpoint_path)
model.eval()

# prediction file 
out_file = f'results/d11948002_hw8_{_exp}.csv'

#%%

anomality = list()
with torch.no_grad():
  for i, data in enumerate(test_dataloader):
        img = data.float().to(device)

        img = img.view(img.shape[0], -1)
        output, latent_vector, reconstruct = model(img)
        #reconstruct = reconstruct.view(reconstruct.shape[0], -1)

        loss = reconstruction_loss(reconstruct, img).sum(-1) 

        anomality.append(loss)
anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['score'])
df.to_csv(out_file, index_label = 'ID')

#%%




