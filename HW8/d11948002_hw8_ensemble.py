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

_exp = 'ensemble'
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
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Linear(256, 64), 
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 256),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(256),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.BatchNorm1d(1024),
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
        self.encoder1 = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 64)
        )
        
        # Second Encoder
        self.encoder2 = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 64)
        )

        self.encoder3 = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.1),
            nn.Linear(256, 64)
        )

        self.encoder4 = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, 64)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.1),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.1),
            nn.Linear(1024, 64 * 64 * 3), 
            nn.Tanh()
        )

    def forward(self, x):
        
        encoded1 = self.encoder1(x)
        encoded2 = self.encoder2(x)
        encoded3 = self.encoder3(x)
        encoded4 = self.encoder4(x)
        
        # Concatenate the encoded features
        encoded = torch.cat((encoded1, encoded2, encoded3, encoded4), dim=1)
        
        # Decode the concatenated features
        decoded = self.decoder(encoded)
        
        return decoded


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
learning_rate = 3e-4

# Build training dataloader
x = torch.from_numpy(train)
train_dataset = CustomTensorDataset(x)

train_sampler = RandomSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size)

# Model
model_type = 'fcn'   # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
model_classes = {'fcn': fcn_autoencoder(), 'cnn': conv_autoencoder(), 'multi': MultiEncoderAutoencoder()}
model = model_classes[model_type].to(device)

# Loss and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

#%%

eval_batch_size = 200

# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
checkpoint_path1 = 'models/best_model_model1.pt'
checkpoint_path2 = 'models/best_model_model2.pt'
checkpoint_path3 = 'models/best_model_model3.pt'
#checkpoint_path4 = 'models/best_model_model4.pt'
model1 = torch.load(checkpoint_path1)
model2 = torch.load(checkpoint_path2)
model3 = torch.load(checkpoint_path3)
#model4 = torch.load(checkpoint_path4)
model.eval()

# prediction file 
out_file = f'results/d11948002_hw8_{_exp}.csv'

#%%

anomality = list()
with torch.no_grad():
  for i, data in enumerate(test_dataloader):
        img = data.float().to(device)

        if model_type in ['fcn','multi']:
            img = img.view(img.shape[0], -1)
        output1 = model1(img)
        output2 = model2(img)
        output3 = model3(img)
        #output4 = model4(img)

        if model_type in ['vae']:
            output = output[0]

        if model_type in ['fcn','multi']:
            loss1 = eval_loss(output1, img).sum(-1)
            loss2 = eval_loss(output2, img).sum(-1)
            loss3 = eval_loss(output3, img).sum(-1)
            #loss4 = eval_loss(output4, img).sum(-1)
            
            loss = (loss1 + loss2 + loss3)/3
            #loss = (loss1 + loss2 + loss3 + loss4)/4
        else:
            loss = eval_loss(output, img).sum([1, 2, 3])

        anomality.append(loss)
anomality = torch.cat(anomality, axis=0)
anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()

df = pd.DataFrame(anomality, columns=['score'])
df.to_csv(out_file, index_label = 'ID')

#%%




