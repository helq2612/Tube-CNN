from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
# from resnext import *
import argparse
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn.functional as F
from tqdm import tqdm
from torch.nn import init
import numpy as np
import math
from torch.optim import lr_scheduler
from jhmdb_dataset import JHMDB_dataloader
from args import data_path, c3d_checkpoint, cfg
from tubeNet import TubeNet
from logger import Logger


# parameters setting, can be introduced by parse later on
device_ids = [0,1]
logger = Logger('./logs/logs518_YoLo_7/')
lr = cfg.TRAIN.LEARNING_RATE
lr = lr * 0.01
total_epoch = 30


# load dataloader
dat, size3datasets, anchors_data =JHMDB_dataloader (data_path)
train = dat['train']
test = dat['test']
print('Training Set size is: {:5d},   Testing Set size is: {:5d}.   '
      'Before DataParallel, the size is{:s}'.format(len(train),  len(test), size3datasets))

# get anchor information from dataset
anchors = torch.from_numpy(anchors_data[0])
all_anchors = torch.from_numpy(anchors_data[1])
inds_inside = torch.from_numpy(anchors_data[2])

# build network, and wrap it into DataParallel constructor
net = TubeNet(anchors, all_anchors, inds_inside)
net.cuda()
net = torch.nn.DataParallel(net, device_ids=device_ids)





# initialize model weights
# for m in net.modules():
#     if isinstance(m, nn.Conv2d):
# #         print(m)
#         m.weight.data.normal_(0.0, 0.02)
    
#         torch.nn.init.xavier_uniform(m.weight)
#         kaiming_normal(m.weight.data)
#         m.weight.data = nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
#     elif isinstance(m, nn.BatchNorm2d):
#         nn.init.constant_(m.weight, 1)
#         nn.init.constant_(m.bias, 0)

## use SGD
# optimizer = optim.SGD(net.parameters(), lr = 0.001, momentum=0.9)
# Decay LR by a factor of 0.1 every 7 epochs
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)



## freeze C3D weight
# for param in net.module.tpn.c3d_part2.parameters():
#     param.requires_grad = False
# for param in net.module.tpn.c3d_part1.parameters():
#     param.requires_grad = False

    
# for param in net.module.tpn1.c3d_part2.parameters():
#     param.requires_grad = False
# for param in net.module.tpn1.c3d_part1.parameters():
#     param.requires_grad = False
    

# for param in net.module.tpn2.c3d_part2.parameters():
#     param.requires_grad = False
# for param in net.module.tpn2.c3d_part1.parameters():
#     param.requires_grad = False

    
# for param in net.module.tpn3.c3d_part2.parameters():
#     param.requires_grad = False
# for param in net.module.tpn3.c3d_part1.parameters():
#     param.requires_grad = False
    
params = []    
for key, value in dict(net.named_parameters()).items():
    if value.requires_grad:
        if 'bias' in key:
            params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
            params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
            
                       
optimizer = torch.optim.Adam(params)
print("Current Learning Rate is {:10f}".format(lr))
optimizer.zero_grad()

# print out model parameter size
pytorch_total_params = sum(p.numel() for p in net.parameters())
pytorch_total_params_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
print("Total Model Parameters are :{:12d}, and trainable parameters are :{:12d}".format(pytorch_total_params,
                                                                                       pytorch_total_params_train))



for epoch in range(total_epoch):
    
    for i_batch, sample_batched in enumerate(train):

        clip_frames, clip_bboxes, clip_indice, clip_labels = sample_batched
        clip_frames = Variable(clip_frames.cuda())
        clip_bboxes = Variable(clip_bboxes.cuda())
        clip_indice = Variable(clip_indice.cuda())
        clip_labels = Variable(clip_labels.cuda())
        
        net.train(True)
        
        loss1, loss2, loss3, loss4, loss5 = net(clip_frames, clip_bboxes, clip_indice, clip_labels, 40)
        loss = loss1.mean() + loss2.mean() + loss3.mean() + loss4.mean() + loss5.mean()

        # backward:
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()

        if (i_batch+1) % 5 == 0:
            print ('Step [{:3d}/{:3d}] Epoch[{:3d}/{:3d}], lr: {:.6f},'
                    ' Loss1: {:.4f}, Loss2: {:.4f},'
                    ' Loss3: {:.4f}, Loss4: {:.4f},'
                    ' Loss5: {:.4f}, Total_loss: {:.4f}'.
                   format(i_batch+1, len(train), epoch, total_epoch, lr,
                          loss1.data.cpu().numpy()[0], 
                          loss2.data.cpu().numpy()[0],
                          loss3.data.cpu().numpy()[0],
                          loss4.data.cpu().numpy()[0],
                          loss5.data.cpu().numpy()[0],
                          loss.data.cpu().numpy()[0]))

            # ================================================================== #
            #                        Tensorboard Logging                         #
            # ================================================================== #

            # 1. Log scalar values (scalar summary)
            info = { 'loss1': loss1.data.cpu().numpy()[0], 
                    'loss2': loss2.data.cpu().numpy()[0], 
                    'loss3': loss3.data.cpu().numpy()[0], 
                    'loss4': loss4.data.cpu().numpy()[0], 
                    'loss5': loss5.data.cpu().numpy()[0],
                    'total_loss':loss.data.cpu().numpy()[0] }

            for tag, value in info.items():
                logger.scalar_summary(tag, value, i_batch+1)

            # 2. Log values and gradients of the parameters (histogram summary)
            for tag, value in net.named_parameters():
                
                tag = tag.replace('.', '/')
                logger.histo_summary(tag, value.data.cpu().numpy(), i_batch+1)
                if value.grad is None:
                    continue
                logger.histo_summary(tag+'/grad', value.grad.data.cpu().numpy(), i_batch+1)


# Save the Model
torch.save(net.state_dict(), 'net_20180518_YoLo5.pk')
