# coding: utf-8
'''
step 1: get the bbox loss (4 coordinates, or x, y, wx, wy) and action loss (binary, is an action or not an action)
step 2: use DP to generate linked tube proposals
step 3: use ToI to generate linked tube proposals
step 4: generate bbox loss and action loss (22 classes) for the whole video


TPN ==> _ProposalLayer
'''

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

# class C3D_Pretrain(nn.Module):
#     def __init__(self):
#         super(C3D_Pretrain, self).__init__()
#         c3d = C3D()
#         c3d.load_state_dict(torch.load(c3d_checkpoint))
#         self.c3d = nn.Sequential(*list(c3d.modules())[:-7])
        
#     def forward(self, inputs):
#         return self.c3d(inputs)


    
    
class TubeNet(nn.Module):
    def __init__(self, anchors, all_anchors, inds_inside,  batchNorm=False):
        super(TubeNet, self).__init__()
        
        # Recognizing Net part, from C3D
#         self.c3d = C3D_Pretrain()
        self.n_classes = 21
        self.tpn = TPN(anchors, all_anchors, inds_inside).cuda(0)
        self.tpn1 = TPN(anchors, all_anchors, inds_inside).cuda(1)
        self.tpn2 = TPN(anchors, all_anchors, inds_inside).cuda(2)
        self.tpn3 = TPN(anchors, all_anchors, inds_inside).cuda(3)
        

        
#         self.grid_size = cfg.POOLING_SIZE * 2 if cfg.CROP_RESIZE_WITH_MAX_POOL else cfg.POOLING_SIZE
        
    def forward(self, clips_frames, clips_bboxes, clips_indice, clips_labels, clip_num):
        '''
        clips_frames: torch.Size([1, 33, 8, 240, 320, 3]) 
        clips_bboxes: torch.Size([1, 33, 8, 4])
        clips_indice: torch.Size([1, 33, 8, 2])
        clips_labels: torch.Size([1, 33, 8])
        
        '''
#         batchId = clips_frames.data(0,0,0,0,0,0)
        clips_num = clips_frames.data.shape[1] # 33
        video = []
        # change from Variable to data
#         clips_frames = clips_frames.data
        clips_bboxes = clips_bboxes.data
        clips_indice = clips_indice.data
        clips_labels = clips_labels.data
        cls_los = 0
        prd_los = 0
        act_los = 0
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0
        count = 0
        GPU = torch.cuda.current_device()
        for clip_idx in range(clips_num):
            # clip has 4 keys: ['clip_idx', 'clip_bboxes', 'clip_frame', 'clip_label']
            one_clip_frames = clips_frames[:, clip_idx, :, :, :,:]
            one_clip_bboxes = clips_bboxes[:, clip_idx, :, :]
            one_clip_labels = clips_labels[:, clip_idx, :]
            
            one_clip_frames = one_clip_frames.cuda(GPU)
            one_clip_bboxes = one_clip_bboxes.cuda(GPU)
            one_clip_labels = one_clip_labels.cuda(GPU)
            
            if GPU == 0:
                loss1, loss2,loss3, loss4, loss5, rois, cls_prob, bbox_pred, rois_label = self.tpn(one_clip_frames, 
                                              one_clip_bboxes,
                                              one_clip_labels)
            elif GPU == 1:
                loss1, loss2,loss3, loss4, loss5, rois, cls_prob, bbox_pred, rois_label  = self.tpn1(one_clip_frames, 
                                               one_clip_bboxes,
                                               one_clip_labels)
            elif GPU == 2:
                loss1, loss2,loss3, loss4, loss5, rois, cls_prob, bbox_pred, rois_label  = self.tpn2(one_clip_frames, 
                                               one_clip_bboxes,
                                               one_clip_labels)
            else:
                loss1, loss2,loss3, loss4, loss5, rois, cls_prob, bbox_pred, rois_label  = self.tpn3(one_clip_frames, 
                                               one_clip_bboxes,
                                               one_clip_labels)
            '''
            If it is in training phrase, use ground truth bbox for refining
            '''

            cls_los += loss1
            prd_los += loss2
            act_los += loss3
            RCNN_loss_cls += loss4
            RCNN_loss_bbox += loss5
            count = clip_idx
            if clip_idx >= clip_num:
                
                break

        return cls_los/count, prd_los/count, act_los/count, RCNN_loss_cls/count, RCNN_loss_bbox/count


        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
        
        