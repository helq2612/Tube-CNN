from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import time
import os
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

from tubeNet import TubeNet
from logger import Logger


dat, size3datasets, anchors_data =JHMDB_dataloader ('/nfs/stak/users/heli/heli/datasets/data/jhmdb')
train = dat['train']
test = dat['test']
print(len(train),  len(test), size3datasets)

anchors = torch.from_numpy(anchors_data[0])
all_anchors = torch.from_numpy(anchors_data[1])
inds_inside = torch.from_numpy(anchors_data[2])
# anchors = torch.from_numpy(anchors_data[0]).cuda()
# all_anchors = torch.from_numpy(anchors_data[1]).cuda()
# inds_inside = torch.from_numpy(anchors_data[2]).cuda()




net = TubeNet(anchors, all_anchors, inds_inside)
net.cuda()
# optimizer = torch.optim.Adam(params)
net = torch.nn.DataParallel(net, device_ids=[0,1])

logger = Logger('./logs/logs518_YoLo_7/')

params = []
from args import c3d_checkpoint, cfg
lr = cfg.TRAIN.LEARNING_RATE




lr = lr * 0.01
print("lr =", lr)

    
    
for key, value in dict(net.named_parameters()).items():
    if value.requires_grad:
        if 'bias' in key:
            params += [{'params':[value],'lr':lr*(cfg.TRAIN.DOUBLE_BIAS + 1), \
                'weight_decay': cfg.TRAIN.BIAS_DECAY and cfg.TRAIN.WEIGHT_DECAY or 0}]
        else:
            params += [{'params':[value],'lr':lr, 'weight_decay': cfg.TRAIN.WEIGHT_DECAY}]
optimizer = torch.optim.Adam(params)
print(lr, optimizer)
optimizer = nn.DataParallel(optimizer, device_ids=[0,1])
optimizer.zero_grad()


total_epoch = 30
for epoch in range(total_epoch):
    for i_batch, sample_batched in enumerate(train):
        print("GPU = ",torch.cuda.current_device())
        clip_frames, clip_bboxes, clip_indice, clip_labels = sample_batched
        clip_frames = Variable(clip_frames.cuda(), requires_grad=True)
        clip_bboxes = Variable(clip_bboxes.cuda(), requires_grad=True)
        clip_indice = Variable(clip_indice.cuda(), requires_grad=True)
        clip_labels = Variable(clip_labels.cuda(), requires_grad=True)
        net.train(True)
        
        loss1, loss2, loss3, loss4, loss5 = net(clip_frames, clip_bboxes, clip_indice, clip_labels, 40)
        print("=======000", loss1)
        loss1 = loss1.mean()
        print("=======///")
        loss2 = loss2.mean()
        loss3 = loss3.mean()
        loss4 = loss4.mean()
        loss5 = loss5.mean()
        print("=======111")
        loss = loss1 + loss2 + loss3 + loss4 + loss5
        
        print("=======222")
#         exp_lr_scheduler.step(epoch)
        # backward:
        optimizer.zero_grad()
        print("=======333",optimizer )
        print("=======333++5, loss = ", loss)
        loss.sum().backward()
#         loss.backward()
        print("=======444")
        optimizer.step()
#         print("At epoch %4d, batch %4d, lr=, the training loss is:%f"%(epoch, i_batch,lr,loss.data.cpu().numpy()[0]))

        if (i_batch+1) % 5 == 0:
            print ('Step [{}/{}] Epoch[{}/{}], lr: {:.6f}, Loss1: {:.4f}, Loss2: {:.4f},  Loss3: {:.4f}, Loss4: {:.4f},  Loss5: {:.4f}, Total_loss: {:.4f}'.
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
