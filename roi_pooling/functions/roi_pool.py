# coding: utf-8
import torch
from torch.autograd import Function
from .._ext import roi_pooling
import pdb

class RoIPoolFunction(Function):
    def __init__(ctx, pooled_height, pooled_width, spatial_scale):
        ctx.pooled_width = pooled_width
        ctx.pooled_height = pooled_height
        ctx.spatial_scale = spatial_scale
        ctx.feature_size = None

    def forward(ctx, features, rois): 
        # ctx is a context object that can be used
        #   to stash information for backward computation
        ctx.feature_size = features.size()   
        # differ from the original one, here the size of ctx.feature_size is:
        #  1x8x240x320x3, so 
        '''
        这里原本的roi pooling输入的是一张图片, 但是tube cnn这里,输入的feature是一个8 frame的cube
        因此, 这里的程序就不做改动了, 但是外面输入的时候, 只选择其中一个frame的
        '''
        
        
        batch_size, num_channels, data_height, data_width = ctx.feature_size
#         batch_size, num_frame, data_height, data_width, num_channels = ctx.feature_size
        num_rois = rois.size(0)
#         print("num_rois", num_rois, num_channels, ctx.pooled_height, ctx.pooled_width)
        output = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_()
        ctx.argmax = features.new(num_rois, num_channels, ctx.pooled_height, ctx.pooled_width).zero_().int()
        ctx.rois = rois
        if not features.is_cuda:
            _features = features.permute(0, 2, 3, 1)
            roi_pooling.roi_pooling_forward(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                            _features, rois, output)
        else:
            roi_pooling.roi_pooling_forward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                                 features, rois, output, ctx.argmax)

        return output

    def backward(ctx, grad_output):
        assert(ctx.feature_size is not None and grad_output.is_cuda)
        batch_size, num_channels, data_height, data_width = ctx.feature_size
        grad_input = grad_output.new(batch_size, num_channels, data_height, data_width).zero_()

        roi_pooling.roi_pooling_backward_cuda(ctx.pooled_height, ctx.pooled_width, ctx.spatial_scale,
                                              grad_output, ctx.rois, grad_input, ctx.argmax)

        return grad_input, None
