#%%
import numpy as np
import torch
import torch.nn as nn
import random
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import gc


#%%

def same_seeds(seed):
    random.seed(seed) 
    np.random.seed(seed)  
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def load_feat(path):
    feat = torch.load(path)
    return feat

def shift(x, n):
    if n < 0:
        left = x[0].repeat(-n, 1)
        right = x[:n]
    elif n > 0:
        right = x[-1].repeat(n, 1)
        left = x[n:]
    else:
        return x

    return torch.cat((left, right), dim=0)

def concat_feat(x, concat_n, model_type='LSTM'):
    assert concat_n % 2 == 1 # n must be odd
    if concat_n < 2:
        return x
    seq_len, feature_dim = x.size(0), x.size(1)
    x = x.repeat(1, concat_n) 
    x = x.view(seq_len, concat_n, feature_dim).permute(1, 0, 2) # concat_n, seq_len, feature_dim
    mid = (concat_n // 2)
    for r_idx in range(1, mid+1):
        x[mid + r_idx, :] = shift(x[mid + r_idx], r_idx)
        x[mid - r_idx, :] = shift(x[mid - r_idx], -r_idx)

    #return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)
    # --- Modify Here to train LSTM ---
    if model_type == 'LSTM':
        return x.permute(1,0,2)
    else:
        return x.permute(1, 0, 2).view(seq_len, concat_n * feature_dim)

def preprocess_data(split, feat_dir, phone_path, concat_nframes, train_ratio=0.8, model_type='LSTM'):
    class_num = 41 # NOTE: pre-computed, should not need change

    if split == 'train' or split == 'val':
        mode = 'train'
    elif split == 'test':
        mode = 'test'
    else:
        raise ValueError('Invalid \'split\' argument for dataset: PhoneDataset!')

    label_dict = {}
    if mode == 'train':
        for line in open(os.path.join(phone_path, f'{mode}_labels.txt')).readlines():
            line = line.strip('\n').split(' ')
            label_dict[line[0]] = [int(p) for p in line[1:]]
        
        # split training and validation data
        usage_list = open(os.path.join(phone_path, 'train_split.txt')).readlines()
        random.shuffle(usage_list)
        train_len = int(len(usage_list) * train_ratio)
        usage_list = usage_list[:train_len] if split == 'train' else usage_list[train_len:]

    elif mode == 'test':
        usage_list = open(os.path.join(phone_path, 'test_split.txt')).readlines()

    usage_list = [line.strip('\n') for line in usage_list]
    print('[Dataset] - # phone classes: ' + str(class_num) + ', number of utterances for ' + split + ': ' + str(len(usage_list)))

    max_len = 3000000
    #X = torch.empty(max_len, 39 * concat_nframes)
    # --- Modify Here to train LSTM ---
    if model_type == 'LSTM':
        X = torch.empty(max_len, concat_nframes, 39)
    else:
        X = torch.empty(max_len, 39 * concat_nframes)

    if mode == 'train':
        y = torch.empty(max_len, dtype=torch.long)

    idx = 0
    for i, fname in tqdm(enumerate(usage_list)):
        feat = load_feat(os.path.join(feat_dir, mode, f'{fname}.pt'))
        cur_len = len(feat)
        feat = concat_feat(feat, concat_nframes, model_type=model_type)
        if mode == 'train':
          label = torch.LongTensor(label_dict[fname])

        X[idx: idx + cur_len, :] = feat
        if mode == 'train':
          y[idx: idx + cur_len] = label

        idx += cur_len

    X = X[:idx, :]
    if mode == 'train':
      y = y[:idx]

    print(f'[INFO] {split} set')
    print(X.shape)
    if mode == 'train':
      print(y.shape)
      return X, y
    else:
      return X


class LibriDataset(Dataset):
    def __init__(self, X, y=None):
        self.data = X
        if y is not None:
            self.label = torch.LongTensor(y)
        else:
            self.label = None

    def __getitem__(self, idx):
        if self.label is not None:
            return self.data[idx], self.label[idx]
        else:
            return self.data[idx]

    def __len__(self):
        return len(self.data)


#%%

class BasicBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BasicBlock, self).__init__()

        # TODO: apply batch normalization and dropout for strong baseline.
        # Reference: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html (batch normalization)
        #       https://pytorch.org/docs/stable/generated/torch.nn.Dropout.html (dropout)
        self.block = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim),
            nn.Dropout(0.75),
        )
        # self.block = nn.LSTM(         
        #             input_size=input_size,
        #             hidden_size=hidden_size,        
        #             num_layers=num_layers,          
        #             batch_first=True,     #(batch, time_step, input_size)
        #             dropout=0.1
        #             )

    def forward(self, x):
        x = self.block(x)
        return x


class Classifier(nn.Module):
    def __init__(self, input_dim, output_dim=41, hidden_layers=1, hidden_dim=256):
        super(Classifier, self).__init__()

        self.fc = nn.Sequential(
                    BasicBlock(input_dim, hidden_dim),
                    *[BasicBlock(hidden_dim, hidden_dim) for _ in range(hidden_layers)],
                    nn.Linear(hidden_dim, output_dim)
                    )

    def forward(self, x):
        x = self.fc(x)
        return x



class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTM, self).__init__()

        self.lstm = nn.LSTM(         
                    input_size=input_size,
                    hidden_size=hidden_size,        
                    num_layers=num_layers,          
                    batch_first=True,     #(batch, time_step, input_size)
                    dropout=0.3,
                    bidirectional = True
                    )

        self.out = nn.Sequential(
                    nn.Linear(hidden_size*2, hidden_size), # multiply 2 because of bidirectional
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size, hidden_size//2), 
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size//2),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size//2, hidden_size//4), 
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size//4),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size//4, hidden_size//8), 
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_size//8),
                    nn.Dropout(0.3),
                    nn.Linear(hidden_size//8, 41)   # 41 is class of output
                )

    def forward(self, x):
        # x.shape (batch, time_step, input_size)
        # output.shape (batch, time_step, output_size)
        # hidden_state.shape (n_layers, batch, hidden_size)
        # cell_state.shape (n_layers, batch, hidden_size)
        output, (hidden_state, cell_state) = self.lstm(x, None)
        out = self.out(output[:, -1, :])
        return out


#%% Hyperparameters Configs

# data prarameters
# TODO: change the value of "concat_nframes" for medium baseline
concat_nframes = 39   # the number of frames to concat with, n must be odd (total 2k+1 = n frames)
train_ratio = 0.7   # the ratio of data used for training, the rest will be used for validation

# training parameters
seed = 52          # random seed
batch_size = 512        # batch size
num_epoch = 25         # the number of training epoch
learning_rate = 3e-3     # learning rate
weight_decay = 0.03
model_path = './model2.ckpt'  # the path where the checkpoint will be saved

# model parameters for Linear Classifer. Used for gradescope.
# TODO: change the value of "hidden_layers" or "hidden_dim" for medium baseline
input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
hidden_layers = 6          # the number of hidden layers
hidden_dim = 128           # the hidden dim

# model parameters for LSTM
input_size = 39 
hidden_size = 512
num_layers = 10

# decide which type of model to use, default: LSTM
model_type = 'LSTM' 

same_seeds(seed)
device = 'cuda:1' if torch.cuda.is_available() else 'cuda:3'
print(f'DEVICE: {device}')

#%%
# preprocess data
train_X, train_y = preprocess_data(split='train', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio, model_type=model_type)
val_X, val_y = preprocess_data(split='val', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, train_ratio=train_ratio, model_type=model_type)

# get dataset
train_set = LibriDataset(train_X, train_y)
val_set = LibriDataset(val_X, val_y)

# remove raw feature to save memory
del train_X, train_y, val_X, val_y
gc.collect()

# get dataloader
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

#%%

# create model, define a loss function, and optimizer
#model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)
if model_type == 'LSTM':
    model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
else:
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)

criterion = nn.CrossEntropyLoss() 
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.5)
print(model)
#%%

best_acc = 0.0
for epoch in range(num_epoch):
    train_acc = 0.0
    train_loss = 0.0
    val_acc = 0.0
    val_loss = 0.0
    
    # training
    model.train() # set the model to training mode
    for i, batch in enumerate(tqdm(train_loader)):
        features, labels = batch
        features = features.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad() 
        outputs = model(features) 
        
        loss = criterion(outputs, labels)
        loss.backward() 
        optimizer.step() 
        
        _, train_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        train_acc += (train_pred.detach() == labels.detach()).sum().item()
        train_loss += loss.item()
    
    # validation
    model.eval() # set the model to evaluation mode
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_loader)):
            features, labels = batch
            features = features.to(device)
            labels = labels.to(device)
            outputs = model(features)
            
            loss = criterion(outputs, labels) 
            
            _, val_pred = torch.max(outputs, 1) 
            val_acc += (val_pred.cpu() == labels.cpu()).sum().item() # get the index of the class with the highest probability
            val_loss += loss.item()

    scheduler.step()
    print(f'[{epoch+1:03d}/{num_epoch:03d}] Train Acc: {train_acc/len(train_set):3.5f} Loss: {train_loss/len(train_loader):3.5f} | Val Acc: {val_acc/len(val_set):3.5f} loss: {val_loss/len(val_loader):3.5f}')

    # if the model improves, save a checkpoint at this epoch
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), model_path)
        print(f'saving model with acc {best_acc/len(val_set):.5f}')
#%%

del train_set, val_set
del train_loader, val_loader
gc.collect()

#%%

# load data
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat', phone_path='./libriphone', concat_nframes=concat_nframes, model_type=model_type)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

# load model
#model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)

if model_type == 'LSTM':
    model = LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers).to(device)
else:
    model = Classifier(input_dim=input_dim, hidden_layers=hidden_layers, hidden_dim=hidden_dim).to(device)

model.load_state_dict(torch.load(model_path))


#%%

pred = np.array([], dtype=np.int32)

model.eval()
with torch.no_grad():
    for i, batch in enumerate(tqdm(test_loader)):
        features = batch
        features = features.to(device)

        outputs = model(features)

        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability
        pred = np.concatenate((pred, test_pred.cpu().numpy()), axis=0)


with open('/home/u/qqaazz800624/2023_Machine_Learning/HW2/outputs/d11948002_hw2_model3.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(pred):
        f.write('{},{}\n'.format(i, y))


#%%

