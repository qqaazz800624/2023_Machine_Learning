#%%

'''
reference link:
1. https://github.com/Singyuan/Machine-Learning-NTUEE-2022/tree/master/hw2
2. https://github.com/Joshuaoneheart/ML2021-HWs
3. https://github.com/pai4451/ML2021

predefined model: efficientnet_b1
'''
#%%


_exp_name = "model10"
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
                transforms.Normalize([0.490, 0.455, 0.405], [0.230, 0.225, 0.225])
                ])

# However, it is also possible to use augmentation in the testing phase.
# You may use train_tfm to produce a variety of images and then test using ensemble methods
#reference link:
#https://github.com/Joshuaoneheart/ML2021-HWs
#https://github.com/pai4451/ML2021
train_tfm = transforms.Compose([
    # Resize the image into a fixed shape (height = width = 128)
    transforms.RandomRotation(30), #對圖片從 (-30,30)之間隨機選擇旋轉角度
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
    transforms.RandomHorizontalFlip(p=0.5), #將一半的圖片進行水平方向翻轉，因為訓練的圖片主要是食物，水平翻轉應該也要能認得出來是什麼食物
    # transforms.RandomResizedCrop((224, 224)),
    # transforms.RandomHorizontalFlip(),
    #ImageNetPolicy(),
    transforms.AutoAugment(),
    transforms.Resize((224, 224)),
    # You may add some transforms here.
    #transforms.RandomHorizontalFlip(p=0.5), #將一半的圖片進行水平方向翻轉，因為訓練的圖片主要是食物，水平翻轉應該也要能認得出來是什麼食物
    #transforms.RandomCrop((128,128), padding = 10),
    # transforms.RandomAffine(degrees=(-20, 20),translate=(0.1, 0.3),scale=(0.5, 0.75)),
    # transforms.ColorJitter(brightness=0.1,contrast=0.2,saturation=0,hue=0),
    # ToTensor() should be the last one of the transforms.
    transforms.ToTensor(),
    transforms.Normalize([0.490, 0.455, 0.405], [0.230, 0.225, 0.225])
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

  
#%% load pretrained model architecture

# "cuda" only when GPUs are available.
device = "cuda:3" if torch.cuda.is_available() else "cuda:0"

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input 維度 [3, 128, 128]
        self.cnn = models.efficientnet_b1(weights=False).to(device)
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


#%%

# "cuda" only when GPUs are available.
#device = "cuda:0" if torch.cuda.is_available() else "cuda:0"

# Initialize a model, and put it on the device specified.
#model = Classifier().to(device)
model = MyModel().to(device)

# The number of batch size.
batch_size = 64

# The number of training epochs.
n_epochs = 200

# If no improvement in 'patience' epochs, early stop.
patience = 30

# For the classification task, we use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, you may fine-tune some hyperparameters such as learning rate on your own.
learning_rate = 3e-4
weight_decay = 0.02
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=20, verbose=False, 
                                                       threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08)
#%%

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
train_set = FoodDataset('/neodata/ML/hw3_dataset/train', tfm=train_tfm)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_set = FoodDataset('/neodata/ML/hw3_dataset/valid', tfm=test_tfm)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

#%%

# Initialize trackers, these are not parameters and should not be changed
stale = 0
best_acc = 0

for epoch in range(n_epochs):

    # ---------- Training ----------
    # Make sure the model is in train mode before training.
    model.train()

    # These are used to record information in training.
    train_loss = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_loss.append(loss.item())
        train_accs.append(acc)
        
    train_loss = sum(train_loss) / len(train_loss)
    train_acc = sum(train_accs) / len(train_accs)
    #scheduler.step()
    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------
    # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        # We don't need gradient in validation.
        # Using torch.no_grad() accelerates the forward process.
        with torch.no_grad():
            logits = model(imgs.to(device))

        # We can still compute the loss (but not the gradient).
        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    scheduler.step(metrics = valid_acc)
    # Print the information.
    print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"/home/u/qqaazz800624/2023_Machine_Learning/HW3/ckpts/{_exp_name}_best.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment in {patience} consecutive epochs, early stopping")
            break


#%%

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
test_set = FoodDataset("/neodata/ML/hw3_dataset/test", tfm=test_tfm)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)


#%%

#model_best = Classifier().to(device)
model_best = MyModel().to(device)
model_best.load_state_dict(torch.load(f"/home/u/qqaazz800624/2023_Machine_Learning/HW3/ckpts/{_exp_name}_best.ckpt"))
model_best.eval()
prediction = []
with torch.no_grad():
    for data,_ in tqdm(test_loader):
        test_pred = model_best(data.to(device))
        test_label = np.argmax(test_pred.cpu().data.numpy(), axis=1)
        prediction += test_label.squeeze().tolist()

#%%

#create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv(f"/home/u/qqaazz800624/2023_Machine_Learning/HW3/outputs/d11948002_hw3_{_exp_name}.csv",index = False)

#%%


