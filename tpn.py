# coding: utf-8
'''
This TPN (tube proposal network) is used to generate tube proposals from given feature maps in a clip of video
Input: 
one 8-frames video clip (batch_size, T, H, W, D) = (1, 8, 300, 400, 3)

Output:
1. some layers in the C3D
    1.1 conv2
    1.2 conv5
  
2. the tube proposals
    2.1 loss1: cls loss of each anchors
    2.2 loss2: bboxes regression loss of each anchors
    2.3 bbox scores
    2.4 bbox coordinates
    
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
from bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch

from _proposalLayer import _ProposalLayer
from anchor_target_layer import _AnchorTargetLayer
from net_utils import _smooth_l1_loss, _smooth_l1_loss1
from cpn import CPN
from proposal_target_layer_cascade import _ProposalTargetLayer
from roi_pooling.modules.roi_pool import _RoIPooling




class TPN(nn.Module):

    def __init__(self, anchors, all_anchors, inds_inside):
        super(TPN, self).__init__()
        # init some para
        self.image_shape = [[240, 320]] # for one batch, TODO: maybe need to change here
        self.anchors = anchors            # (630, x, y, xw, yw)                anchors coordinates
        self.inds_inside = inds_inside
        self.all_anchors =  all_anchors
        # get C3D part, use pretrained weight
        c3d = C3D()
        
        c3d.load_state_dict(torch.load(c3d_checkpoint))
        self.c3d_part1 = nn.Sequential(*list(c3d.modules())[1:4])        # be careful about these two indices
                                                                        # get conv2 
        self.c3d_part2 = nn.Sequential(*list(c3d.modules())[4:13])      # 
        
        self.BN1 = torch.nn.BatchNorm2d(512)
        # 
        # for RPN
        self._CPN = CPN(self.anchors, all_anchors, inds_inside)
        
        self.n_classes = 22
        
        self.RCNN_proposal_target = _ProposalTargetLayer(self.n_classes)
        self.RCNN_roi_pool = _RoIPooling(cfg.POOLING_SIZE, cfg.POOLING_SIZE, 1.0/16.0)
        
        self.head_to_tail_ =  torch.nn.Sequential(
            nn.Linear(512 * 7 * 7, 1024),   # change from 4096 to 2048, for memory limit
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(1024, 4096),          # change from 4096 to 2048, for memory limit
            nn.ReLU(True)
        )
        
        self.RCNN_bbox_pred = torch.nn.Linear(4096, 4 * self.n_classes) 
        self.RCNN_cls_score = torch.nn.Linear(4096, self.n_classes)
        
        
    
    def forward(self, inputs, gt_boxes, gt_labels):
        # inputs: 1x8x240x320x3
        # gt_boxes: 1x8x4
        # gt_labels: 1
        '''
        Step 1: generate two output feature maps from C3D
        '''

        batch_size = inputs.size(0)

        output_conv2 = self.c3d_part1(Variable(inputs.permute(0, 4, 1,2,3)))  # torch.Size([1, 128, 8, 120, 160])
        output_conv5 = self.c3d_part2(output_conv2)                 # torch.Size([1, 512, 1, 15, 20])

        output_conv5 = output_conv5.view(batch_size, 512, 15, 20)   # torch.Size([1, 512, 15, 20])
        output_conv5 = self.BN1(output_conv5)
        rpn_loss_cls, rpn_loss_box, rpn_loss_action, rois =self._CPN(output_conv5, gt_boxes.contiguous(), gt_labels)
        gt_boxes = gt_boxes[:,0,:].view(1, 1, 5)
        if self.training:
            roi_data = self.RCNN_proposal_target(rois, gt_boxes)
            rois, rois_label, rois_target, rois_inside_ws, rois_outside_ws = roi_data

            rois_label = Variable(rois_label.view(-1).long())
            rois_target = Variable(rois_target.view(-1, rois_target.size(2)))
            rois_inside_ws = Variable(rois_inside_ws.view(-1, rois_inside_ws.size(2)))
            rois_outside_ws = Variable(rois_outside_ws.view(-1, rois_outside_ws.size(2)))
        else:
            rois_label = None
            rois_target = None
            rois_inside_ws = None
            rois_outside_ws = None
            rpn_loss_cls = 0
            rpn_loss_bbox = 0
        rois = Variable(rois)
        

        pooled_feat = self.RCNN_roi_pool(output_conv5, rois.view(-1,5))
        
        pool5 = pooled_feat.view(pooled_feat.size(0), -1)
        
        pooled_feat = self.head_to_tail_(pool5)
        
        # compute bbox offset
        bbox_pred = self.RCNN_bbox_pred(pooled_feat)
        
        if self.training:
            # bbox_pred.shape = (128L, 88L)
            bbox_pred_view = bbox_pred.view(bbox_pred.size(0), int(bbox_pred.size(1) / 4), 4)
            
            # bbox_pred_view.shape =  (128L, 22L, 4L)
            bbox_pred_select = torch.gather(bbox_pred_view, 
                                            1, 
                                            rois_label.view(rois_label.size(0), 
                                                            1, 
                                                            1).expand(rois_label.size(0), 
                                                                      1, 
                                                                      4))
            bbox_pred = bbox_pred_select.squeeze(1) 

        # compute object classification probability
        # pooled_feat.shape = (128L, 4096L)
        cls_score = self.RCNN_cls_score(pooled_feat)
        
        # cls_score.shape = (128L, 22L)
        cls_prob = F.softmax(cls_score, dim = 1)
        # cls_prob.shape = (128L, 22L)
        
        RCNN_loss_cls = 0
        RCNN_loss_bbox = 0

        if self.training:
            # classification loss
            # cls_score.shape = (128L, 22L), rois_label.shape = (128L,)
            RCNN_loss_cls = F.cross_entropy(cls_score, rois_label)
            
            # bounding box regression L1 loss
            # rois_target.shape = (128L, 4L), rois_inside_ws.shape = (128L, 4L), rois_outside_ws.shape =(128L, 4L)
            # bbox_pred.shape = (128L, 4L)
            RCNN_loss_bbox = _smooth_l1_loss(bbox_pred, rois_target, rois_inside_ws, rois_outside_ws)
            # RCNN_loss_bbox.shape = (1L,)
            
        # rois.shape = (1L, 128L, 5L)
        cls_prob = cls_prob.view(batch_size, rois.size(1), -1)
        bbox_pred = bbox_pred.view(batch_size, rois.size(1), -1)
        
        # more: cls_prob and bbox_pred for this clip       
        return rpn_loss_cls, rpn_loss_box, rpn_loss_action, RCNN_loss_cls, RCNN_loss_bbox, rois, cls_prob, bbox_pred, rois_label

    
    @staticmethod
    def reshape(x, d):
        input_shape = x.size()
        x = x.view(
            input_shape[0],
            int(d),
            int(float(input_shape[1] * input_shape[2]) / float(d)),
            input_shape[3]
        )
        return x