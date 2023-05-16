#%%

import torch
import torch.nn as nn
from pytorchcv.model_provider import get_model as ptcv_get_model
import random
import numpy as np

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cuda:2')
batch_size = 8

def same_seeds(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	
same_seeds(0) 

#%%

# the mean and std are the calculated statistics from cifar_10 dataset
cifar_10_mean = (0.491, 0.482, 0.447) # mean for the three channels of cifar_10 images
cifar_10_std = (0.202, 0.199, 0.201) # std for the three channels of cifar_10 images

# convert mean and std to 3-dimensional tensors for future operations
mean = torch.tensor(cifar_10_mean).to(device).view(3, 1, 1)
std = torch.tensor(cifar_10_std).to(device).view(3, 1, 1)

epsilon = 8/255/std

#%%

root = './data' # directory for storing benign images
# benign images: images which do not contain adversarial perturbations
# adversarial images: images which include adversarial perturbations

#%%

import os
import glob
import shutil
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import Dataset, DataLoader

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_10_mean, cifar_10_std)
])

class AdvDataset(Dataset):
    def __init__(self, data_dir, transform):
        self.images = []
        self.labels = []
        self.names = []
        '''
        data_dir
        ├── class_dir
        │   ├── class1.png
        │   ├── ...
        │   ├── class20.png
        '''
        for i, class_dir in enumerate(sorted(glob.glob(f'{data_dir}/*'))):
            images = sorted(glob.glob(f'{class_dir}/*'))
            self.images += images
            self.labels += ([i] * len(images))
            self.names += [os.path.relpath(imgs, data_dir) for imgs in images]
        self.transform = transform
    def __getitem__(self, idx):
        image = self.transform(Image.open(self.images[idx]))
        label = self.labels[idx]
        return image, label
    def __getname__(self):
        return self.names
    def __len__(self):
        return len(self.images)

adv_set = AdvDataset(root, transform=transform)
adv_names = adv_set.__getname__()
adv_loader = DataLoader(adv_set, batch_size=batch_size, shuffle=False)

print(f'number of images = {adv_set.__len__()}')

#%%

# to evaluate the performance of model on benign images
def epoch_benign(model, loader, loss_fn):
    model.eval()
    train_acc, train_loss = 0.0, 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        yp = model(x)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
    return train_acc / len(loader.dataset), train_loss / len(loader.dataset)

#%%

#input diversity

#references: https://github.com/Singyuan/Machine-Learning-NTUEE-2022/blob/master/hw10/hw10.ipynb
#references: https://adversarial-attacks-pytorch.readthedocs.io/en/latest/_modules/torchattacks/attacks/difgsm.html
#MI-FGSM: https://arxiv.org/pdf/1710.06081.pdf
#DIM: https://arxiv.org/pdf/1803.06978.pdf

import torch.nn.functional as F
torch.manual_seed(100)

def input_diversity(x, resize_rate=0.9, diversity_prob=0.5):
    img_size = x.shape[-1]
    img_resize = int(img_size * resize_rate)

    if resize_rate < 1:
        img_size = img_resize
        img_resize = x.shape[-1]

    rnd = torch.randint(low=img_size, high=img_resize, size=(1,), dtype=torch.int32)
    rescaled = F.interpolate(x, size=[rnd, rnd], mode='bilinear', align_corners=False)
    h_rem = img_resize - rnd
    w_rem = img_resize - rnd
    pad_top = torch.randint(low=0, high=h_rem.item(), size=(1,), dtype=torch.int32)
    pad_bottom = h_rem - pad_top
    pad_left = torch.randint(low=0, high=w_rem.item(), size=(1,), dtype=torch.int32)
    pad_right = w_rem - pad_left

    padded = F.pad(rescaled, [pad_left.item(), pad_right.item(), pad_top.item(), pad_bottom.item()], value=0)

    return padded if torch.rand(1) < diversity_prob else x


#%%

# perform fgsm attack
def fgsm(model, x, y, loss_fn, epsilon=epsilon):
    x_adv = x.detach().clone() # initialize x_adv as original benign image x
    x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
    loss = loss_fn(model(x_adv), y) # calculate loss
    loss.backward() # calculate gradient
    # fgsm: use gradient ascent on x_adv to maximize loss
    grad = x_adv.grad.detach()
    x_adv = x_adv + epsilon * grad.sign()
    return x_adv

# alpha and num_iter can be decided by yourself
alpha = 0.8/255/std

#references: https://github.com/Singyuan/Machine-Learning-NTUEE-2022/blob/master/hw10/hw10.ipynb
#references: https://github.com/pai4451/ML2021/blob/main/hw10/hw10_adversarial_attack.ipynb
def ifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    x_adv = x
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        # x_adv = fgsm(model, x_adv, y, loss_fn, alpha) # call fgsm with (epsilon = alpha) to obtain new x_adv
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
        loss = loss_fn(model(x_adv), y) # calculate loss
        loss.backward() # calculate gradient
        # fgsm: use gradient ascent on x_adv to maximize loss
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()

        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv


alpha = 2/255/std

#references: https://github.com/Singyuan/Machine-Learning-NTUEE-2022/blob/master/hw10/hw10.ipynb
#references: https://github.com/pai4451/ML2021/blob/main/hw10/hw10_adversarial_attack.ipynb
def difgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20):
    x_adv = x
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        # x_adv = fgsm(model, x_adv, y, loss_fn, alpha) # call fgsm with (epsilon = alpha) to obtain new x_adv
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
        # loss = loss_fn(model(x_adv), y) # calculate loss
        
        # ===== Here replace by input_diversity =====
        loss = loss_fn(model(input_diversity(x_adv)), y) # calculate loss

        loss.backward() # calculate gradient
        # fgsm: use gradient ascent on x_adv to maximize loss
        grad = x_adv.grad.detach()
        x_adv = x_adv + alpha * grad.sign()

        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv

#references: https://github.com/Singyuan/Machine-Learning-NTUEE-2022/blob/master/hw10/hw10.ipynb
#references: https://github.com/pai4451/ML2021/blob/main/hw10/hw10_adversarial_attack.ipynb
def mifgsm(model, x, y, loss_fn, epsilon=epsilon, alpha=alpha, num_iter=20, decay=0.0):
    x_adv = x
    # initialze momentum tensor
    momentum = torch.zeros_like(x).detach().to(device)
    # write a loop of num_iter to represent the iterative times
    for i in range(num_iter):
        x_adv = x_adv.detach().clone()
        x_adv.requires_grad = True # need to obtain gradient of x_adv, thus set required grad
        loss = loss_fn(model(input_diversity(x_adv)), y) # calculate loss
        loss.backward() # calculate gradient
        # TODO: Momentum calculation
        
        grad = x_adv.grad.detach()
        grad = grad/torch.mean(torch.abs(grad), dim=(1,2,3), keepdim=True)
        grad = grad + momentum*decay
        momentum = grad.clone() # original velocity

        x_adv = x_adv + alpha * grad.sign()
        x_adv = torch.max(torch.min(x_adv, x+epsilon), x-epsilon) # clip new x_adv back to [x-epsilon, x+epsilon]
    return x_adv


#%%

# perform adversarial attack and generate adversarial examples
def gen_adv_examples(model, loader, attack, loss_fn):
    model.eval()
    adv_names = []
    train_acc, train_loss = 0.0, 0.0
    for i, (x, y) in enumerate(loader):
        x, y = x.to(device), y.to(device)
        x_adv = attack(model, x, y, loss_fn) # obtain adversarial examples
        yp = model(x_adv)
        loss = loss_fn(yp, y)
        train_acc += (yp.argmax(dim=1) == y).sum().item()
        train_loss += loss.item() * x.shape[0]
        # store adversarial examples
        adv_ex = ((x_adv) * std + mean).clamp(0, 1) # to 0-1 scale
        adv_ex = (adv_ex * 255).clamp(0, 255) # 0-255 scale
        adv_ex = adv_ex.detach().cpu().data.numpy().round() # round to remove decimal part
        adv_ex = adv_ex.transpose((0, 2, 3, 1)) # transpose (bs, C, H, W) back to (bs, H, W, C)
        adv_examples = adv_ex if i == 0 else np.r_[adv_examples, adv_ex]
    return adv_examples, train_acc / len(loader.dataset), train_loss / len(loader.dataset)

# create directory which stores adversarial examples
def create_dir(data_dir, adv_dir, adv_examples, adv_names):
    if os.path.exists(adv_dir) is not True:
        _ = shutil.copytree(data_dir, adv_dir)
    for example, name in zip(adv_examples, adv_names):
        im = Image.fromarray(example.astype(np.uint8)) # image pixel value should be unsigned int
        im.save(os.path.join(adv_dir, name))


#%%

################ BOSS BASELINE ######################

class ensembleNet(nn.Module):
    def __init__(self, model_names):
        super().__init__()
        self.models = nn.ModuleList([ptcv_get_model(name, pretrained=True) for name in model_names])
        
    def forward(self, x):
        #################### TODO: boss baseline ###################
        paras = []
        for model in self.models:
            tmp = model(x.clone())
            tmp = tmp.view(tmp.size(0), -1)
            paras.append(tmp)

        result = torch.zeros(batch_size, 10).to(device)
        for para in paras:
            result += para/len(self.models)

        return result


#%%


from pytorchcv.model_provider import get_model as ptcv_get_model

model_names = ['wrn40_8_cifar10',
               'preresnet110_cifar10',
               'resnet110_cifar10',
               'resnet56_cifar10',
               'resnext29_16x64d_cifar10',
               'resnet1202_cifar10',
               'diapreresnet110_cifar10',
               'densenet40_k24_bc_cifar10',
               'densenet100_k12_cifar10',
               'sepreresnet56_cifar10',
               'sepreresnet110_cifar10',
               'ror3_110_cifar10',
               'rir_cifar10',
               'shakeshakeresnet20_2x16d_cifar10',
               'seresnet110_cifar10',
               'diaresnet56_cifar10',
               'diaresnet110_cifar10']


ensemble_model = ensembleNet(model_names).to(device)
ensemble_model.eval()
# Create ensemble model
loss_fn = nn.CrossEntropyLoss()
print('After Ensemble')
benign_acc, benign_loss = epoch_benign(ensemble_model, adv_loader, loss_fn)
print(f'benign_acc = {benign_acc:.5f}, benign_loss = {benign_loss:.5f}')



#%%
# FGSM

adv_examples, fgsm_acc, fgsm_loss = gen_adv_examples(ensemble_model, adv_loader, fgsm, loss_fn)
print(f'fgsm_acc = {fgsm_acc:.5f}, fgsm_loss = {fgsm_loss:.5f}')

create_dir(root, 'fgsm', adv_examples, adv_names)


#%% 
# I-FGSM

adv_examples, ifgsm_acc, ifgsm_loss = gen_adv_examples(ensemble_model, adv_loader, ifgsm, loss_fn)
print(f'ifgsm_acc = {ifgsm_acc:.5f}, ifgsm_loss = {ifgsm_loss:.5f}')

create_dir(root, 'ifgsm', adv_examples, adv_names)

#%%
# DI-FGSM

adv_examples, difgsm_acc, difgsm_loss = gen_adv_examples(ensemble_model, adv_loader, difgsm, loss_fn)
print(f'difgsm_acc = {difgsm_acc:.5f}, difgsm_loss = {difgsm_loss:.5f}')

create_dir(root, 'difgsm', adv_examples, adv_names)


#%%

'''

%cd fgsm
!tar zcvf ../fgsm.tgz *
%cd ..


%cd difgsm
!tar zcvf ../difgsm.tgz *
%cd ..

'''


#%%


#%%





#%%




#%%






#%%




#%%




#%%







#%%




#%%




#%%







#%%




#%%




#%%