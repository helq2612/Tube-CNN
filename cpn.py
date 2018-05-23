# coding: utf-8

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
from net_utils import _smooth_l1_loss



class CPN(nn.Module):
    def __init__(self, anchors, all_anchors, inds_inside):
        super(CPN, self).__init__()
        self.image_shape = [[240, 320]] # for one batch, TODO: maybe need to change here
        self.anchors = anchors            # (630, x, y, xw, yw)                anchors coordinates
        self.inds_inside = inds_inside
        self.all_anchors =  all_anchors
        self.nc_score_out = 2 * 12    # 2(bg/fg)  * 12 (anchors)
        self.nc_bbox_out  = 4 * 12    # 4(coords) * 12 (anchors)
        c3d = C3D()
        self.action_num = 22 #-1       # 21(classes) do not consider bg
        self.action_anchor_num = self.action_num * 12
        c3d.load_state_dict(torch.load(c3d_checkpoint))

        self.RPN_Conv = nn.Conv2d(512, 512, 3, 1, 1, bias=True)
        self.BN1 = nn.BatchNorm2d(512)
        self.RPN_cls_bbox_action = nn.Conv2d(512, 
                                             self.nc_score_out + self.nc_bbox_out +self.action_anchor_num , 
                                             1, 1, 0)
        self.BN2 = nn.BatchNorm2d(self.nc_score_out + self.nc_bbox_out +self.action_anchor_num)
        self.RPN_proposal = _ProposalLayer(self.anchors, self.all_anchors)
        self.RPN_anchor_target = _AnchorTargetLayer(self.anchors, self.inds_inside)
        
        
        
    def forward(self, inputs, gt_boxes_input, gt_labels):
        batch_size =  inputs.size(0)
        ''''
        Step 2: generate rpn class score probability for each pixels on the feature map
        '''
        gt_boxes = gt_boxes_input[:,:,:4].contiguous()
        gt_label = gt_boxes_input[0,0,4] #- 1# here do not consider bg
        output_conv5 = self.RPN_Conv(inputs)
        output_conv5 = self.BN1(output_conv5)
        output_conv5 = F.relu(output_conv5, inplace=True)
        # output size: (batch_size, channels, H, W) = (1, 12*2, 15, 20)  ==>checked
        rpn_cls_bbox_action = self.RPN_cls_bbox_action(output_conv5)    # generate action score for two class
        rpn_cls_bbox_action = self.BN2(rpn_cls_bbox_action)
        rpn_cls_score = rpn_cls_bbox_action[:,:self.nc_score_out,:,:]
        # output size: (batch_size, channels, H, W) = (1, 2, 24*15/2, 20) = (1, 2, 228, 20)
        rpn_cls_score_reshape = self.reshape(rpn_cls_score, 2)
#         print("rpn_cls_score_reshape.shape = ", rpn_cls_score_reshape.shape)
        # output size: the same as above
        rpn_cls_prob_reshape = F.softmax(rpn_cls_score_reshape, dim=1)
        # output size: (batch_size, channels, H, W) = (1, 24, 15, 20) 
        rpn_cls_prob = self.reshape(rpn_cls_prob_reshape, self.nc_score_out) # for 12 anchors
        # this output channel num = 12* 4 = 48, the same feature map space 19*20
        # so batchsize, 48, 15, 20
        '''
        Step 3: generate bounding box proposals for each pixel on the feature map
        '''
        # output size: (batch_size, channels, H, W) = (1, 12*4, 15, 20) ==>checked 
        rpn_bbox_pred = rpn_cls_bbox_action[:,
                                            self.nc_score_out: (self.nc_score_out + self.nc_bbox_out),
                                            :,:]
        arg_key = 'TRAIN' if self.training else 'TEST'
        rois = self.RPN_proposal((rpn_cls_prob.data, rpn_bbox_pred.data, self.image_shape, arg_key))
        '''
        
        1. 先把rois得到的BBOXes mapping 回conv2, 8个frame的, 这个就是tube proposals, 也是需要output出来的
        2. 然后把
        '''
        # generate tube proposals
        # conv5_feat_cube, conv2_feat_tube = self.mapback(rois, output_conv2)
        '''
        Step 4: get action score for each pixels
        '''
        rpn_action_score = rpn_cls_bbox_action[:,
                                            (self.nc_score_out + self.nc_bbox_out):,
                                            :,:]
        
        self.rpn_loss_cls = 0
        self.rpn_loss_box = 0
        self.rpn_loss_action = 0
        if self.training:
            assert gt_boxes is not None
            '''
            Step 5: compare the anchors and bounding box predicted (from step 3), get the
            IoU value for each anchors. Filtering them by the threshold             
            '''
            rpn_data = self.RPN_anchor_target((rpn_cls_score.data, gt_boxes, self.image_shape))

            rpn_cls_score_reshaped = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(batch_size, -1, 2)
                
            rpn_label = rpn_data[0].view(batch_size, -1)
            test_label_1 = torch.sum((rpn_label == 1).int(), 1)
            test_label_0 = torch.sum((rpn_label == 0).int(), 1)
                
            rpn_keep2 = Variable(rpn_label.view(-1).ne(1).ne(1).nonzero().view(-1))
            rpn_keep = Variable(rpn_label.view(-1).ne(-1).nonzero().view(-1))               
                
            rpn_cls_score_reshaped = torch.index_select(rpn_cls_score_reshaped.view(-1,2), 0, rpn_keep) 
                
            rpn_label = torch.index_select(rpn_label.view(-1), 0, rpn_keep.data)
                
            rpn_label = Variable(rpn_label.long())
            '''
            Step 6: use Cross-Entropy to calculate the classification loss
            '''
            #a = F.cross_entropy(rpn_cls_score_reshaped, rpn_label)
            self.rpn_loss_cls += F.cross_entropy(rpn_cls_score_reshaped, rpn_label)
            #print("self.rpn_loss_cls.shape = ", self.rpn_loss_cls.shape)
            #print("self.rpn_loss_cls  = ", self.rpn_loss_cls)
            # 计算regress loss
            fg_cnt = torch.sum(rpn_label.data.ne(0))
            rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
            # compute bbox regression loss
            rpn_bbox_inside_weights = Variable(rpn_bbox_inside_weights)
            rpn_bbox_outside_weights = Variable(rpn_bbox_outside_weights)
            rpn_bbox_targets = Variable(rpn_bbox_targets)
            '''
            Step 7: use Smooth L1 to calculate the regression loss of four coordinates
            '''

            self.rpn_loss_box += _smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights,
                                                                rpn_bbox_outside_weights, sigma=3, dim=[1,2,3])
                                 
            '''
            Step 8: calculate the action score loss
            1. input is batch_size * action_num * H * w, action_num = 21, do not include background
            2. select bboxes that only fg, this will result to only 5 bboxes
            3. calculate loss
            '''

            rpn_action_score = torch.sigmoid(rpn_action_score) # get sigmoid // try softmax later
            rpn_action_score = rpn_action_score.view(batch_size, 15, 20, 12,  self.action_num)
            rpn_action_score = rpn_action_score.permute(1, 2, 3, 0, 4).view(3600, batch_size, self.action_num).contiguous()
            
            rpn_action_score = torch.index_select(rpn_action_score, 0, rpn_keep2)
            rpn_action_score = rpn_action_score.view(-1, self.action_num)
            gt_action_label = torch.ones(rpn_action_score.size(0)) * gt_label
            gt_action_label = gt_action_label.view(rpn_action_score.size(0))

            gt_action_label = Variable(gt_action_label.cuda().long())
                           
            self.rpn_loss_action += F.cross_entropy(rpn_action_score, gt_action_label)   
        return self.rpn_loss_cls/8, self.rpn_loss_box/8, self.rpn_loss_action, rois
    
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