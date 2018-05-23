# coding: utf-8
'''
_proposalLayer.py
Used by tpn.py

generate the proposal layer, will return:
region of proposals, which will be used in next

init: anchors

self.forward:
input: 
1. rpn_cls_prob.data,   : class information on each pixel, learned from conv
2. rps_bbox_pred.data,  : bbox information on each pixel, learned from conv  
3. self.image_shape,    : original image size
3. arg_key              : 'TRAIN' or 'TEST

这里还必须把RPN的东西从2D转到3D, 因为training的时候,其实是比照着原来的输入图片的.
对于每个clip而言,一共有8个frame, 因此必须把这个2D的BBOX相关的先mapping回3D, 然后才能进行loss的计算

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
from bbox_transform import bbox_transform_inv, clip_boxes, clip_boxes_batch, bbox_transform_inv2
from nms.nms_wrapper import nms


class _ProposalLayer(nn.Module):
    def __init__(self, anchors, all_anchors):
        super(_ProposalLayer, self).__init__()

#         self._anchors = torch.from_numpy(anchors)
        self._anchors = anchors
        self.all_anchors = all_anchors
        self.all_num_anchors = self.all_anchors.size(0)  # for 240 * 320 image, there are 650 anchors for this dataset.
        self._num_anchors = self._anchors.size(0)
        self._num_anchors_type = 12                # 12 anchors
        
    def forward(self, input):
        # input[0]: (batch_size, channels, H, W) = (1, 24, 19, 20)
        # input[1]: (batch_size, channels, H, W) = (1, 12*4, 19, 20)
        # input[2]: (batch_size, H, W) = (1, 240, 320)
        # input[3]: "TEST" or "TRAIN"
        
        all_anchors = self.all_anchors.cuda()
        
        scores      = input[0][:, self._num_anchors_type:, :, :]     # class score (binary) for each feature map pixel
        bbox_deltas = input[1]     # bbox for each feature map pixel, size (batch_size, 48, 19, 20)
        im_info     = input[2]     # image shape, for jhmdb, it is [[240, 320]] TODO1: change this to [240, 320]
        cfg_key     = input[3]     # TRAIN or TEST
        im_info     = np.array(im_info)
        
        pre_nms_topN  = cfg[cfg_key].RPN_PRE_NMS_TOP_N    # train: 12000, test: 6000
        post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N   # train: 2000,  test: 300
        nms_thresh    = cfg[cfg_key].RPN_NMS_THRESH       # train: 0.7,   test: 0.7
        min_size      = cfg[cfg_key].RPN_MIN_SIZE         # train: 8,     test: 16

        batch_size = bbox_deltas.size(0)                  # mostly 1
        
        # since the anchors are obtained from dataset, we can just use it, change it to
        # (batch_size, 3600, 4) TODO: this is different from origin
        all_anchors = all_anchors.contiguous()
        all_anchors = all_anchors.view(1, self.all_num_anchors, 4).expand(batch_size, self.all_num_anchors, 4)
        
        # Transpose and reshape predicted bbox transformations to get them
        # into the same order as the anchors:  change to (batch_size, 19, 20, 48)
        bbox_deltas = bbox_deltas.permute(0, 2, 3, 1).contiguous()
        bbox_deltas = bbox_deltas.view(batch_size, -1, 4)
        
        # Same story for the scores:
        # batch_size, 19, 25, 12
        scores = scores.permute(0, 2, 3, 1).contiguous()
        '''
        x = torch.randn(5, 4)
        print(x.stride(), x.is_contiguous())
        print(x.t().stride(), x.t().is_contiguous())
        x.view(4, 5) # ok
        x.t().view(4, 5) # fails
        '''
        scores = scores.view(batch_size, -1)
        
        # Convert anchors into proposals via bbox transformations
        # so we get a big list of bbox
        ## slide anchors on each pixels on the feature map 19*20, get bounding boxes
        # achors, 630 * 4, means 630 anchors, with 4 coordinates
        #  bbox_deltas, batch_size, 19, 20, 48.  48 means 4 cooridnates * 12 anchors
        # all_anchors.shape = 1x3600x4, bbox_deltas.shape=1x3600x4
        proposals = bbox_transform_inv2(all_anchors, bbox_deltas, batch_size)
        
        # 2. clip predicted boxes to image: TODO: this line is useless, since our input anchor is already fixed with
        # image size.
        ## remove the bboxes that outside of the image boundary
        # proposals.shape = [1, 3600, 4]), im_info = [[240, 320]]
        proposals = clip_boxes(proposals, im_info, batch_size)
        
        
        
        scores_keep = scores         #(batch_size, 12, 19, 25)
        proposals_keep = proposals
        
        _, order = torch.sort(scores_keep, 1, True)   # sort 12 'anchors', here is the cnn output score
        
        output = scores.new(batch_size, post_nms_topN, 5).zero_()
        for i in range(batch_size):
            # # 3. remove predicted boxes with either height or width < threshold
            # # (NOTE: convert min_size to input image scale stored in im_info[2])
            proposals_single = proposals_keep[i] # for one batch, all the anchors for each feature map pixel
                                                 # size: (12, 19, 25)
            scores_single = scores_keep[i]       # binary class score for each feature map pixel
                                                 # size: (12, 19, 25)

            # # 4. sort all (proposal, score) pairs by score from highest to lowest
            # # 5. take top pre_nms_topN (e.g. 6000)
            order_single = order[i]

            if pre_nms_topN > 0 and pre_nms_topN < scores_keep.numel():
                order_single = order_single[:pre_nms_topN]

            proposals_single = proposals_single[order_single, :]
            scores_single = scores_single[order_single].view(-1,1)

            # 6. apply nms (e.g. threshold = 0.7)
            # 7. take after_nms_topN (e.g. 300)
            # 8. return the top proposals (-> RoIs top)

            keep_idx_i = nms(torch.cat((proposals_single, scores_single), 1), nms_thresh, force_cpu=not cfg.USE_GPU_NMS)
            keep_idx_i = keep_idx_i.long().view(-1)

            if post_nms_topN > 0:
                keep_idx_i = keep_idx_i[:post_nms_topN]
            proposals_single = proposals_single[keep_idx_i, :]
            scores_single = scores_single[keep_idx_i, :]

            # padding 0 at the end.
            num_proposal = proposals_single.size(0)
            output[i,:,0] = i
            output[i,:num_proposal,1:] = proposals_single

        return output
    
    
    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass