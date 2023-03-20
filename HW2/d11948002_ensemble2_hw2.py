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
    # reference link:
    # 1. https://github.com/Singyuan/Machine-Learning-NTUEE-2022/tree/master/hw2
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
    # reference link:
    # 1. https://github.com/Singyuan/Machine-Learning-NTUEE-2022/tree/master/hw2
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

class LSTM1(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTM1, self).__init__()

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


class LSTM2(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTM2, self).__init__()

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

class LSTM3(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTM3, self).__init__()

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
    
class LSTM4(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=1):
        super(LSTM4, self).__init__()

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
seed = 6          # random seed
batch_size = 512        # batch size

model1_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW2/model.ckpt'  
model2_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW2/model2.ckpt'  
model3_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW2/model3.ckpt' 
model4_path = '/home/u/qqaazz800624/2023_Machine_Learning/HW2/model4.ckpt'  

# TODO: change the value of "hidden_layers" or "hidden_dim" for medium baseline
input_dim = 39 * concat_nframes  # the input dim of the model, you should not change the value
hidden_layers = 6          # the number of hidden layers
hidden_dim = 128           # the hidden dim

# model parameters for LSTM
model_type='LSTM'

same_seeds(seed)
device = 'cuda:3' if torch.cuda.is_available() else 'cuda:3'
print(f'DEVICE: {device}')

#%%

# load data
test_X = preprocess_data(split='test', feat_dir='./libriphone/feat'
                         , phone_path='./libriphone'
                         , concat_nframes=concat_nframes
                         , model_type=model_type)
test_set = LibriDataset(test_X, None)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

#%%
# load model
model1 = LSTM1(input_size=39, hidden_size=512, num_layers=10).to(device)
model1.load_state_dict(torch.load(model1_path))
model2 = LSTM2(input_size=39, hidden_size=512, num_layers=10).to(device)
model2.load_state_dict(torch.load(model2_path))
model3 = LSTM3(input_size=39, hidden_size=512, num_layers=10).to(device)
model3.load_state_dict(torch.load(model3_path))
model4 = LSTM4(input_size=39, hidden_size=512, num_layers=10).to(device)
model4.load_state_dict(torch.load(model4_path))


#%%

predict = []
model1.eval() # set model1 to evaluation mode
model2.eval() # set model2 to evaluation mode
model3.eval() # set model3 to evaluation mode
model4.eval() # set model4 to evaluation mode

with torch.no_grad():
    for i, data in enumerate(test_loader):
        inputs = data
        inputs = inputs.to(device)
        outputs1 = model1(inputs)
        outputs2 = model2(inputs)
        outputs3 = model3(inputs)
        outputs4 = model4(inputs)
        outputs = (outputs1 + outputs2 + outputs3 + outputs4) / 4
        _, test_pred = torch.max(outputs, 1) # get the index of the class with the highest probability

        for y in test_pred.cpu().numpy():
            predict.append(y)

#%%

with open('/home/u/qqaazz800624/2023_Machine_Learning/HW2/outputs/ensemble2.csv', 'w') as f:
    f.write('Id,Class\n')
    for i, y in enumerate(predict):
        f.write('{},{}\n'.format(i, y))


#%%




#%%