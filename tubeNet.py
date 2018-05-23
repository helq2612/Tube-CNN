# coding: utf-8
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
from tqdm import tqdm
import math
import torch.nn.functional as F
from torch.nn import init
import numpy as np
from C3D_model import C3D
from args import c3d_checkpoint, cfg

from tpn import TPN

from bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch

from proposal_target_layer_cascade import _ProposalTargetLayer
from roi_pooling.modules.roi_pool import _RoIPooling
    
    
class TubeNet(nn.Module):
    def __init__(self, anchors, all_anchors, inds_inside,  batchNorm=False):
        super(TubeNet, self).__init__()

        self.n_classes = 21      
        self.tpn = TPN(anchors, all_anchors, inds_inside)

        
    def forward(self, clips_frames, clips_bboxes, clips_indice, clips_labels, clip_num):
  
        clips_num = clips_frames.data.shape[1] # 33
        video = []
        # change from Variable to data
        clips_frames = clips_frames.data
        clips_bboxes = clips_bboxes.data
        clips_indice = clips_indice.data
        clips_labels = clips_labels.data
        cls_los = 0
        prd_los = 0
        act_los = 0
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        count = 0
        for clip_idx in range(clips_num):
            one_clip_frames = clips_frames[:, clip_idx, :, :, :,:]
            one_clip_bboxes = clips_bboxes[:, clip_idx, :, :]
            one_clip_labels = clips_labels[:, clip_idx, :]

            loss1, loss2,loss3, loss4, loss5, rois, cls_prob, bbox_pred, rois_label = self.tpn(one_clip_frames, 
                                              one_clip_bboxes,
                                              one_clip_labels)

            cls_los += loss1
            prd_los += loss2
            act_los += loss3
            RCNN_loss_cls += loss4
            RCNN_loss_bbox += loss5
            count = clip_idx
            if clip_idx >= clip_num:          # safe to remove this line      
                break
        return cls_los/count, prd_los/count, act_los/count, RCNN_loss_cls/count, RCNN_loss_bbox/count


        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        