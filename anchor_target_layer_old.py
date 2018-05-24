# coding: utf-8
from __future__ import absolute_import

import torch
import torch.nn as nn
import numpy as np
import numpy.random as npr


from args import cfg
from bbox_transform import clip_boxes, bbox_overlaps_batch2, bbox_transform_batch

import pdb

DEBUG = False

try:
    long        # Python 2
except NameError:
    long = int  # Python 3


class _AnchorTargetLayer(nn.Module):
    """
 
    """
    def __init__(self, anchors, inds_inside, num_anchors_type=12):
        super(_AnchorTargetLayer, self).__init__()
        self._anchors = anchors
        self.num_anchors = len(self._anchors) # 630
        self.num_anchors_type = num_anchors_type # 12
        self.num_anchors_total = self.num_anchors_type * (240/16) * (320/16)
        self.total_anchors = self.num_anchors_type * (240/16) * (320/16) # total anchors before consider over boundary
        self.inds_inside = inds_inside # size(0) = 630
        
    def forward(self, input):
        '''
        input[0]: rpn_cls_score.data  -- 1x12*2x15x20
        input[1]: gt_boxes            -- 1x8x4          In paper, it says 
        input[2]: self.image_shape    -- [[240, 320]] 
        '''
        rpn_cls_score = input[0]
        gt_boxes = input[1]
        im_info  = input[2]
        A = self.num_anchors_type
        # feature map shape
        height, width = rpn_cls_score.size(2), rpn_cls_score.size(3)
        
        batch_size = gt_boxes.size(0) # 1
        # 这里630个anchors已经做好了,不需要再计算, 并且每个anchor对应的index也都存在inds_inside里了
        anchors = self._anchors
        
        # label: 1 is positive, 0 is negative, -1 is dont care
        # 这里建立的labels等三个tensor,维度都与conv5后conv2的 输出的相匹配, 但先考虑跟anchor象对应的630个
        labels =               gt_boxes.new(batch_size, self.num_anchors).fill_(-1)
        bbox_inside_weights  = gt_boxes.new(batch_size, self.num_anchors).zero_()
        bbox_outside_weights = gt_boxes.new(batch_size, self.num_anchors).zero_()
        
        overlaps = bbox_overlaps_batch2(anchors * 16, gt_boxes) # map anchors to original input image size, 1x630x8
        max_overlaps, argmax_overlaps = torch.max(overlaps, 2) # along dim 2, means max for all 8 frames 1x630
        # max_overlaps 返回最大的值
        # argmax_overlaps 返回那个frame的gt_boxes 取得, 因此是 0~7 直接的数
        # 两个都是1x630
        
        gt_max_overlaps, _ = torch.max(overlaps, 1)            # along dim 2, means for all 8 frames 1x8
        # 1x8:  0.8816  0.9009  0.9009  0.9009  0.9009  0.8745  0.8644  0.8962
        
        if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0  # <0.3
            
        gt_max_overlaps[gt_max_overlaps==0] = 1e-5
        keep = torch.sum(overlaps.eq(gt_max_overlaps.view(batch_size,1,-1).expand_as(overlaps)), 2)
        # #457的anchor与第一个frame的gtbox一致, #463 anchor与其他7个frame的gtbox一致, 因此就保留了两个anchor
        # 其实就是用来找最大的overlap, 即: 第一个frame中的gt_box, 是落在了457号上, 第2~8个frame上的gtbox,是落在了第463号上
        
        
        if torch.sum(keep) > 0:  # 会找到几个不等于0的,本例子中sum的结果是8=1+7
            labels[keep>0] = 1
        # 到这里为止, 找到了与gt_box最大匹配的anchor

        # fg label: above threshold IOU
        labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1   # >=0.7
        
        
        if cfg.TRAIN.RPN_CLOBBER_POSITIVES:                          # False
            labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

        num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)  # 0.5 * 256 =128

        sum_fg = torch.sum((labels == 1).int(), 1)    # positive有10个
        sum_bg = torch.sum((labels == 0).int(), 1)    # negative有467个, dont care有153 个
        
        for i in range(batch_size):
            # subsample positive labels if we have too many
            if sum_fg[i] > num_fg:
                fg_inds = torch.nonzero(lables[i] == 1).view(-1)
                rand_num = torch.from_numpy(np.random.permutation(fg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = fg_inds[rand_num[:fg_inds.size(0)-num_fg]]
                labels[i][disable_inds] = -1
                
            num_bg = cfg.TRAIN.RPN_BATCHSIZE - sum_fg[i]  # 256-10 = 246, 下面这行重复了
            #num_bg = cfg.TRAIN.RPN_BATCHSIZE - torch.sum((labels == 1).int(), 1)[i]

            # subsample negative labels if we have too many
            if sum_bg[i] > num_bg:      # 467 > 246
                bg_inds = torch.nonzero(labels[i] == 0).view(-1) #一个467长度的list, 表示467个index
                #rand_num = torch.randperm(bg_inds.size(0)).type_as(gt_boxes).long()

                rand_num = torch.from_numpy(np.random.permutation(bg_inds.size(0))).type_as(gt_boxes).long()
                disable_inds = bg_inds[rand_num[:bg_inds.size(0)-num_bg]]
                labels[i][disable_inds] = -1 #强行将467-246=221个bg的扔掉
            
        offset = torch.arange(0, batch_size)*gt_boxes.size(1)

        argmax_overlaps = argmax_overlaps + offset.view(batch_size, 1).type_as(argmax_overlaps)
        bbox_targets = _compute_targets_batch(anchors, 
                                              gt_boxes.view(-1,4)[argmax_overlaps.view(-1), :].view(batch_size, -1, 4))
        '''Compute bounding-box regression targets for an image.'''
        
        
        # use a single value instead of 4 values for easy index.
        bbox_inside_weights[labels==1] = cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS[0]   #1.0

        if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:         # RPN_POSITIVE_WEIGHT = -1
            num_examples = torch.sum(labels >= 0)     # 447
            positive_weights = 1.0 / num_examples     # 1 / 447
            negative_weights = 1.0 / num_examples
        else:
            assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                    (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))

        bbox_outside_weights[labels == 1] = positive_weights  # 将这个1/447 赋值给这两种label
        bbox_outside_weights[labels == 0] = negative_weights
        
        # labels.dim()=2
        labels = _unmap(labels, self.num_anchors_total, self.inds_inside, batch_size, fill=-1)
        bbox_targets = _unmap(bbox_targets, self.num_anchors_total, self.inds_inside, batch_size, fill=0)
        bbox_inside_weights = _unmap(bbox_inside_weights, self.num_anchors_total, self.inds_inside, batch_size, fill=0)
        bbox_outside_weights = _unmap(bbox_outside_weights, self.num_anchors_total, self.inds_inside, batch_size, fill=0)

        outputs = []

        labels = labels.view(batch_size, height, width, A).permute(0,3,1,2).contiguous()
        labels = labels.view(batch_size, 1, A * height, width)
        outputs.append(labels)          # labels

        bbox_targets = bbox_targets.view(batch_size, height, width, A*4).permute(0,3,1,2).contiguous()
        outputs.append(bbox_targets)    # bbox

        anchors_count = bbox_inside_weights.size(1)
        bbox_inside_weights = bbox_inside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)

        bbox_inside_weights = bbox_inside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()

        outputs.append(bbox_inside_weights)

        bbox_outside_weights = bbox_outside_weights.view(batch_size,anchors_count,1).expand(batch_size, anchors_count, 4)
        bbox_outside_weights = bbox_outside_weights.contiguous().view(batch_size, height, width, 4*A)\
                            .permute(0,3,1,2).contiguous()
        outputs.append(bbox_outside_weights)

        return outputs
        
        
        

def _unmap(data, count, inds, batch_size, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """

    if data.dim() == 2:
        ret = torch.Tensor(batch_size, count).fill_(fill).type_as(data)
        ret[:, inds] = data
    else:
        ret = torch.Tensor(batch_size, count, data.size(2)).fill_(fill).type_as(data)
        ret[:, inds,:] = data
    return ret


def _compute_targets_batch(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""

    return bbox_transform_batch(ex_rois, gt_rois[:, :, :4])
        
        
        
        
        