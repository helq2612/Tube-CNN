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
    

问题1: 没法保持住clips之间的关系. 原文只是用了overlap score来强行链接. 但是, 因为input的clip数量是不一定的(所以后面用ToI pooling来进行维度统一), 这样并没有产生一些列的TPN来对应这一系列的clips. 也就是说, 这个TPN不但需要记忆本段clip的内容,还需要记录跟别的clip段的关系. 这明显是不可能的(原文中).
问题2: 产生的bbox跟tuble proposals 都只是应对两个class(是, 或者不是). 因此, 这个video中只能有一个action class
问题3: 产生的那些anchors, 并不能直接进行loss处理, 因为这些anchors需要mapping回到之前的8个frame. 因为只有之前那8个frame上才有真的BBox.

结论: 先把 Tube CNN 的前面这一部分做完吧, 不然没法做下去了
中文的步骤说明:
1. 输入为一个clip的video, 先产生conv2 和conv5
2. conv5 传入两个conv, 得到feature map上,每个pixel的 score, bbox
3. 由training data经过Kmean 得到一些列anchor, 这些anchor是对应了原图/feature map而言的
4. 由BBOX 跟anchor, 可以得到一些列的适配了的BBOX, 并且都对应了有一个score. 到这一步为止, 还没有考虑任何的loss --- 周六上午完成!
5. 由上一步的适配过的BBOX, 根据conv2 mapping回去, 得到对应8个frame的bbox, 可以假设为一个tube. 这就是paper中提到的tube proposal.
6. 得到之后, 再组成一个新的网络, 进行loss的训练
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
    '''
    tube proposal network for each chip
    inputs:
    anchors:        12 types of anchors for this dataset, used k mean method to get
                    size: (batch_size, anchor_num, num_coordinates) = (1, 630, 4)
                    here 630 is varied for different input image sizes
    feature maps 1: from conv5b
                    size: (batch_size, channels, T, H, W) = (1, 512, 1, 19, 25)
    feature maps 2: from conv2
                    size: (batch_size, channels, T, H, W) = (1, 128, 8, 150, 200)                
    outputs:
    conv5 output: yes need, for total network
    conv5 feature cube: no need
    conv2 feature cube: no need
    loss part:
    '''
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
        # one for conv2: (batch_size, channels, T, H, W) = (1, 128, 8, 120, 160)   3D conv
        # one for conv5: (batch_size, channels, H, W) = (1, 512, 15, 20)           2D conv
#         print(" inside of tpn.py==>",inputs.shape) #torch.Size([1, 8, 240, 320, 3])
        batch_size = inputs.size(0)
#         print("batch size = ", batch_size)
#         print("inside of tpn, batchsize = ", batch_size)
#         print("inputs.shape=", inputs.shape, "inputs[0,0,0,0,0]=",inputs[0,0,0,0,0])
        output_conv2 = self.c3d_part1(inputs.permute(0, 4, 1,2,3))  # torch.Size([1, 128, 8, 120, 160])
        output_conv5 = self.c3d_part2(output_conv2)                 # torch.Size([1, 512, 1, 15, 20])
#         test = self.test(output_conv5)
        output_conv5 = output_conv5.view(batch_size, 512, 15, 20)   # torch.Size([1, 512, 15, 20])
        output_conv5 = self.BN1(output_conv5)
        rpn_loss_cls, rpn_loss_box, rpn_loss_action, rois =self._CPN(output_conv5, gt_boxes.contiguous(), gt_labels)
#         print("gt_boxes.shape = ", gt_boxes.shape)
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
        
        
#         print("EEROR HERE: rois_label = ", rois_label, "rois_target", rois_target, "rois_inside_ws:", rois_inside_ws, "rois_outside_ws", rois_outside_ws)
#         print("ERROR END")
        
        '''
        下面需要计算pred 的BBoxes了, 以及pred_class
        差不多直接套用faster rcnn的代码
        
        '''
        
        # only consider Pool:
        # output_conv5.shape = (1L, 512L, 15L, 20L)  rois.view(-1,5).shape = (128L, 5L), 
        pooled_feat = self.RCNN_roi_pool(output_conv5, rois.view(-1,5))
        
        #  pooled_feat.shape = (128L, 512L, 7L, 7L)
        pool5 = pooled_feat.view(pooled_feat.size(0), -1)
        
        #  pool5.shape =  (128L, 25088L)
        pooled_feat = self.head_to_tail_(pool5)
        
        # compute bbox offset
        # pooled_feat.shape = (128L, 4096L)
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
            # bbox_pred_select.shape = (128L, 1L, 4L)
            
        
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
    
#     def _head_to_tail(self, pool5):
#         '''
#         这是原code中的vgg.classifier, 他这里不取最后的Dropout layer, 
#             nn.Linear(512 * 7 * 7, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(),
        
#         '''
        
        
#         pool5_flat = pool5.view(pool5.size(0), -1)
#         fc7 = self.RCNN_top(pool5_flat)

#         return fc7
    
    
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