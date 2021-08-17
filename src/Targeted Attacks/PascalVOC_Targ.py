import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision
from tqdm.auto import tqdm
from torchvision.utils import make_grid
from torchvision import models
from torchvision import datasets, transforms as T

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
%matplotlib inline
from sklearn.metrics import accuracy_score
import numpy as np
from utils import pgd_targ, IoUAcc, decode_segmap

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
torch.cuda.empty_cache()

batch_size = 4
img_size = (520, 520) # original input size to the model is (520,520) but all images in dataset are of different sizes

trans = T.Compose([T.Resize(img_size),T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
dataset = datasets.VOCSegmentation(r'/home/shk/PascalVOC', year = '2012',
                                   image_set = 'trainval',download = False, transform = trans,
                                   target_transform = T.Resize(img_size), transforms = None)# change path for local use

X, y, yrep = [], [], []
for i in range(batch_size):
    num = torch.randint(0,1449,(1,1)).item()
    X.append(dataset[num][0])
    y.append(np.asarray(dataset[num][1]))
    yrep.append(dataset[num][1])
X, y = torch.stack(X), torch.tensor(y).unsqueeze(1)

class_names = {0:'background', 1:'aeroplane', 2:'bicycle', 3:'bird', 4:'boat', 5:'bottle', 6:'bus', 7:'car', 8:'cat',
               9:'chair', 10:'cow', 11:'diningtable', 12:'dog', 13:'horse', 14:'motorbike', 15:'person', 16:'pottedplant',
               17:'sheep', 18:'sofa', 19:'train', 20:'tvmonitor', 255: 'void'}

net = models.segmentation.deeplabv3_resnet101(pretrained=True, num_classes=21, aux_loss=None).eval()

yp = net(X)['out']
m = torch.softmax(yp, 1)
pred = torch.argmax(m, 1)
IoUAcc(y, pred)

delta = pgd_targ(net, X, y, epsilon=0.10, alpha=100, num_iter = 15, y_targ = y[1] ) # Second image in the batch is targeted
ypa = net((X.float()+ delta.float()))['out']
n = torch.softmax(ypa, 1)
preda = torch.argmax(n,1)
IoUa, Acca = IoUAcc(y, preda)


