# Please note that this file is to be run in the scaleatt folder of the repo https://github.com/EQuiw/2019-scalingattack to be functional.
#The concept of scaling the image is directly taken from this repo.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
from torchvision import datasets, transforms as T
from utils import result attack_image, IoUAcc, decode_segmap

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline
from sklearn.metrics import accuracy_score
import numpy as np

# Image scaling attack library (https://github.com/EQuiw/2019-scalingattack)
# Note: File is to be created in scaleatt folder to comply with the following requirements
from utils.plot_image_utils import plot_images_in_actual_size
from scaling.ScalingGenerator import ScalingGenerator
from scaling.SuppScalingLibraries import SuppScalingLibraries
from scaling.SuppScalingAlgorithms import SuppScalingAlgorithms
from scaling.ScalingApproach import ScalingApproach
from attack.QuadrScaleAttack import QuadraticScaleAttack
from attack.direct_attacks.DirectNearestScaleAttack import DirectNearestScaleAttack
from attack.ScaleAttackStrategy import ScaleAttackStrategy


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

batch_size = 4
img_size = (520, 520) # original input size to the model is (520,520) but all images in dataset are of different sizes
trans = T.Compose([T.Resize(img_size), T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
#T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
dataset = datasets.VOCSegmentation(r'/home/shk/PascalVOC/', year = '2012',
                                   image_set = 'trainval',download = False, transform = trans,
                                   target_transform = T.Resize(img_size), transforms = None)
X, y, yrep = [], [], []
for i in range(batch_size):
    num = torch.randint(0,1449,(1,1)).item()
    X.append(dataset[num][0])
    y.append(np.asarray(dataset[num][1]))
    yrep.append(dataset[num][1])
X, y = torch.stack(X), torch.tensor(y)

net = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None).eval()

yp = net(X)['out']
yp.size()
pred = torch.argmax(yp,1)
IoUAcc(y, pred)

tran = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
x = torch.zeros((X.size(0), 3, 130, 130))
X_targ = torch.zeros_like(X)
delta_v, delta_h = [], []
inp = int(input('Which image to target (1,2,3,4): '))
for i in range(X.size(0)):
    att_img, lower, deltav_, deltah_ = result_attack_image(X[i], X[inp-1])
    delta_v.append(deltav_)
    delta_h.append(deltah_)
    print('Attack image received')
    X_targ[i] = tran(att_img)
    x[i] = tran(lower)

    
ypa = net(x)['out']
high = torch.nn.functional.interpolate(ypa, size = img_size, mode='nearest')
preda = torch.argmax(high,1)
preda.unique()
IoUAcc(y,preda)
