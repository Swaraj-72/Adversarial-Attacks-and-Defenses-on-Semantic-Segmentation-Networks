import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from torchvision import models
from torchvision import datasets, transforms as T
from utils import decode_segmap, IoUAcc

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline
from sklearn.metrics import accuracy_score
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

batch_size = 4

img_size = (520,520) # original input size to the model is (520,520) but all images in dataset are of different sizes in PascalVOC

trans = T.Compose([T.Resize(img_size),T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

dataset = datasets.VOCSegmentation(r'/datasets/PascalVOC/....', year = '2012', image_set = 'val',download = False, transform = trans,
                                   target_transform = T.Resize(img_size), transforms = None) # Path to be updated for local use.

X, y, yrep = [], [], []
for i in range(batch_size):
    num = torch.randint(0,1449,(1,1)).item()
    X.append(dataset[num][0])
    y.append(np.asarray(dataset[num][1]))
    yrep.append(dataset[num][1])
X, y = torch.stack(X), torch.tensor(y).unsqueeze(1)

#print(X.size(), y.size())

net = models.segmentation.deeplabv3_resnet101(pretrained=True, progress=True, num_classes=21, aux_loss=None).eval() #Any pre-trained model from pytorch can be made used of.

class_names = {1:'background', 2:'aeroplane', 3:'bicycle', 4:'bird', 5:'boat', 6:'bottle', 7:'bus', 8:'car', 9:'cat',
               10:'chair', 11:'cow', 12:'diningtable', 13:'dog', 14:'horse', 15:'motorbike', 16:'person', 17:'pottedplant',
               18:'sheep', 19:'sofa', 20:'train', 21:'tvmonitor' }

yp = net(X)['out']
m = torch.softmax(yp,1)
pred = torch.argmax(m,1)
IoUAcc(y, pred)

delta1 = pgd(net, X, y, epsilon=0.10, alpha=1e2, num_iter=10) # Various values of epsilon, alpha can be used to play with.
ypa1 = net((X.float()+ delta1.float()))['out']
n = torch.softmax(ypa1,1) 
preda1 = torch.argmax(n,1)
IoUa1, Acca1 = IoUAcc(y, preda1)
