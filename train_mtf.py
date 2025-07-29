'''
训练测试代码
'''

from swin_MTFNet import Dataset_3DCNN, Swin_MTFNet
from swin_MTFNet import train_model

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import numpy as np
from timm.models.layers import DropPath, trunc_normal_
from torch.utils.data import Subset

import torch.distributed as dist

from functools import reduce, lru_cache
from operator import mul
from einops import rearrange
from thop import profile


import logging
# from mmcv.utils import get_logger
# from mmcv.runner import load_checkpoint

import os
from PIL import Image
from torch.utils import data
from sklearn.model_selection import KFold, StratifiedKFold

from torch.nn import init
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import cv2
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from matplotlib import pylab
from sklearn.metrics import f1_score,precision_score,recall_score,roc_auc_score,roc_curve,confusion_matrix,accuracy_score,log_loss
from sklearn.metrics import precision_recall_curve, average_precision_score,roc_curve, auc, confusion_matrix, accuracy_score, classification_report

from torch import einsum
from einops import rearrange, repeat

import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import copy


# Depression=A   Healthy=B


data_path = '/yourpath'

checkpoint_path = '/yourpath'
MAX_frame = 30

img_height = 224
img_width = 224
# epoch = 5
batch_size = 64

transform = transforms.Compose([transforms.Resize([img_height, img_width]),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5], std=[0.5])])

use_cuda = torch.cuda.is_available()                   # check if GPU exists
device = torch.device("cuda:0" if use_cuda else "cpu")   # use CPU or GPU

params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}
params_test = {'batch_size': batch_size, 'shuffle': False, 'num_workers': 0, 'pin_memory': True} if use_cuda else {}

# train_set, valid_set = Dataset_3DCNN(train_path, transform=transform), \
#                        Dataset_3DCNN(validate_path, transform=transform)
# test_set = Dataset_3DCNN(test_path, transform=transform)
data_set = Dataset_3DCNN(data_path, transform=transform)





skf = StratifiedKFold(n_splits=10, shuffle=True)
X = []
y = []
for i in range(len(data_set)):
    Xi, yi = data_set[i]
    X.append(Xi)
    y.append(yi)

X = torch.stack(X)
y = torch.stack(y).squeeze()
count = 0
name = ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]



for train_index, test_index in skf.split(X, y):
    swin_3D = Swin_MTFNet()
    swin_3D = swin_3D.to(device)
    swin_3D = nn.DataParallel(swin_3D)
    flops, params = profile(swin_3D)
    print('flops:', flops, 'params:', params)

    optimizer_ft = optim.SGD(swin_3D.parameters(), lr=0.01, momentum=0.9)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.5)

    print(f"第{count}次分层训练：")
    print('train_index:%s , test_index:%s' % (train_index, test_index))
    path = name[count]
    if not os.path.exists("/yourpath" + path):
        os.makedirs("/yourpath" + path)
    count += 1
    train_subset = Subset(data_set, train_index)
    test_subset = Subset(data_set, test_index)

    dataset_sizes = {'train': len(train_subset), 'val': len(test_subset)}

    train_loader = data.DataLoader(train_subset, **params)
    test_loader = data.DataLoader(test_subset, **params)

    swin_3D = train_model(swin_3D, optimizer_ft, exp_lr_scheduler, train_loader, test_loader, dataset_sizes, path, num_epochs=25)