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


class ResNetEncoder(nn.Module):
    def __init__(self, encoding_dim):
        super(ResNetEncoder, self).__init__()

        resnet = models.resnet18(pretrained=True)
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        self.fc = nn.Linear(resnet.fc.in_features, encoding_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        encoded = self.fc(x)
        return encoded


class Autoencoder(nn.Module):
    def __init__(self, encoding_dim):
        super(Autoencoder, self).__init__()

        self.encoder = ResNetEncoder(encoding_dim)
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 64 * 64 * 3),
            nn.Sigmoid()
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


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
num_epochs = 240
batch_size = 256 # Hint: batch size may be lower
learning_rate = 8e-4
num_workers = 2

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size
                              , num_workers=num_workers, pin_memory=True, drop_last=True)

encoding_dim = 64*64*3
# Model
model_type = 'resnet'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model_classes = {'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'resnet': Autoencoder(encoding_dim=encoding_dim)}
model = model_classes[model_type].to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts

total_steps = len(train_dataloader) * num_epochs
scheduler = StepLR(optimizer, step_size=25, gamma=0.95)


#%%

best_loss = np.inf
model.train()

qqdm_train = qqdm(range(num_epochs), desc=format_str('bold', 'Description'))
for epoch in qqdm_train:
    tot_loss = list()
    for data in train_dataloader:

        # ===================loading=====================
        img = data.float().to(device)
        if model_type in ['fcn']:
            img = img.view(img.shape[0], -1)
        # ===================forward=====================
        output = model(img)
        loss = criterion(output, img)

        tot_loss.append(loss.item())
        # ===================backward====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    # ===================save_best====================
    mean_loss = np.mean(tot_loss)
    if mean_loss < best_loss:
        best_loss = mean_loss
        torch.save(model, f'models/best_model_{_exp}.pt')
    # ===================log========================
    qqdm_train.set_infos({
        'epoch': f'{epoch + 1:.0f}/{num_epochs:.0f}',
        'loss': f'{mean_loss:.4f}',
    })
    # ===================save_last========================
    torch.save(model, f'models/last_model_{_exp}.pt')
    print(f"LR: {scheduler.get_last_lr()[0]}")

#%%

eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
checkpoint_path = f'models/last_model_{_exp}.pt'
model = torch.load(checkpoint_path)
model.eval()

# prediction file 
out_file = f'results/d11948002_hw8_{_exp}.csv'

#%%

anomality = list()
with torch.no_grad():
  for i, data in enumerate(test_dataloader):
        img = data.float().to(device)

        if model_type in ['fcn']:
            img = img.view(img.shape[0], -1)
        output = model(img)

        if model_type in ['vae']:
            output = output[0]
        if model_type in ['fcn']:
            loss = eval_loss(output, img).sum(-1)
        else:
            loss = eval_loss(output, img).sum([1, 2, 3])
        anomality.append(loss)
anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['score'])
df.to_csv(out_file, index_label = 'ID')

#%%




