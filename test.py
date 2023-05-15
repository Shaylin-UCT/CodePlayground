import matplotlib.pyplot as plt
import numpy as np
import torch
import os
from PIL import Image
import torch.nn as nn
 
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from torch.nn import Module, Sequential, Conv2d, ConvTranspose2d, LeakyReLU, BatchNorm2d, ReLU, Tanh, Sigmoid, BCELoss
# path to the image directory
dir_data  = "ImagesforResearch/Elbow"
 
img_dim = 256
# setting image shape to 64x64
img_shape = (img_dim,img_dim, 3)
 
# listing out all file names
nm_imgs   = np.sort(os.listdir(dir_data))
print(nm_imgs)

X_train = []
for file in nm_imgs:
    try:
        img = Image.open(dir_data+'/'+file)
        img = img.resize((img_dim,img_dim))
        img = np.asarray(img)/255
        X_train.append(img)
    except:
        print("something went wrong")
 
X_train = np.array(X_train)
class ElbowFacesDataset(Dataset):
    """Human Faces dataset."""
 
    def __init__(self, npz_imgs):
        """
        Args:
            npz_imgs (string): npz file with all the images (created in gan.ipynb)
        """
        self.imgs = npz_imgs
 
    def __len__(self):
        return len(self.imgs)
 
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image = self.imgs[idx]
        return image
# Preparing dataloader for training
 
transpose_imgs = np.transpose( # imp step to convert image size from (7312, 32,32,3) to (7312, 3,32,32)
    np.float32(X_train), # imp step to convert double -&gt; float (by default numpy input uses double as data type)
    (0, 3,1,2) # tuple to describe how to rearrange the dimensions
    ) 
 
dset = ElbowFacesDataset(transpose_imgs) # passing the npz variable to the constructor class
batch_size = 32
shuffle = True
 
dataloader = DataLoader(dataset = dset, batch_size = batch_size, shuffle = shuffle)


examples = next(iter(dataloader))

for label, img  in enumerate(examples):
   plt.imshow(img.permute(1,2,0))
   plt.show()
   print(f"Label: {label}")