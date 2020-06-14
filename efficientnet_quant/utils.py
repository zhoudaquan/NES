"""
This file contains helper functions for building the model and for loading model parameters.
These helper functions are built to mirror those in the official TensorFlow implementation.
"""

import re
import numpy as np
import pdb
from torch.nn.parameter import Parameter

import math
import collections
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo
from distiller.modules import EltwiseAdd, EltwiseMult, EltwiseDiv


########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate',])


# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])


# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


def relu_fn(x):
    """ Swish activation function """
    return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype)  # uniform [0,1)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


class Conv2dSamePadding_v0(nn.Conv2d):
    """ 2D Convolutions like TensorFlow """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True, nbit=4):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]]*2

        self.num_filters = in_channels
        self.nbits = nbit
        self.ceil = math.ceil
        bit_dims = [nbit, in_channels]
        # self.add = EltwiseAdd()
        # self.div = EltwiseDiv()
        # self.mul = EltwiseMult()
        # #if pad_h > 0 or pad_w > 0:
        # self.static_padding = nn.ZeroPad2d((0 // 2, 0 - 0 // 2, 0 // 2, 0 - 0 // 2))
        # self.basis = Parameter(torch.Tensor(nbit, in_channels), requires_grad=True)

    # def QuantizedWeight(self, x, n, nbit=4):
    #     """
    #     Quantize weight.
    #     Args:
    #         x (tf.Tensor): a 4D tensor.
    #             Must have known number of channels, but can have other unknown dimensions.
    #         name (str): operator's name.
    #         n (int or double): variance of weight initialization.
    #         nbit (int): number of bits of quantized weight. Defaults to 2.
    #     Returns:
    #         tf.Tensor with attribute `variables`.
    #     Variable Names:
    #     * ``basis``: basis of quantized weight.
    #     Note:
    #         About multi-GPU training: moving averages across GPUs are not aggregated.
    #         Batch statistics are computed by main training tower. This is consistent with most frameworks.
    #     """
    #     NORM_PPF_0_75 = 0.6745
    #     MOVING_AVERAGES_FACTOR = 0.9
    #     EPS = 0.0001
    #     # num_filters = x.get_shape().as_list()[-1]
    #     num_filters = x.shape[0]
    #     # initialization of quantization levels, i/(2**nbits)
    #     init_basis = []
    #     base = NORM_PPF_0_75 * ((2. / n) ** 0.5) / (2 ** (nbit - 1))
    #     for j in range(nbit):
    #         init_basis.append([(2 ** j) * base for i in range(num_filters)])
    #     # convert to tf variable, not necessary for pytorch
    #     # init_basis = tf.constant_initializer(init_basis)
    #     # bit_dims = [nbit, num_filters]
    #     num_levels = 2 ** nbit
    #     delta = EPS
    #     # initialize level multiplier
    #     init_level_multiplier = []
    #     for i in range(num_levels):
    #         level_multiplier_i = [0. for j in range(nbit)]
    #         level_number = i
    #         for j in range(nbit):
    #             binary_code = level_number % 2
    #             if binary_code == 0:
    #                 binary_code = -1
    #             level_multiplier_i[j] = float(binary_code)
    #             level_number = level_number // 2
    #         init_level_multiplier.append(level_multiplier_i)
    #     # initialize threshold multiplier
    #     init_thrs_multiplier = []
    #     for i in range(1, num_levels):
    #         thrs_multiplier_i = [0. for j in range(num_levels)]
    #         thrs_multiplier_i[i - 1] = 0.5
    #         thrs_multiplier_i[i] = 0.5
    #         init_thrs_multiplier.append(thrs_multiplier_i)

    #     level_codes = torch.Tensor(init_level_multiplier)
    #     thrs_multiplier = torch.Tensor(init_thrs_multiplier)
        
    #     sum_multiplier= torch.Tensor(np.ones((1, torch.reshape(x, [-1, num_filters]).shape[0])))
    #     sum_multiplier_basis = torch.Tensor(np.ones([1, nbit]))

    #     # NOTE:I think this is not needed for putorch. check it!
    #     # ctx = get_current_tower_context()  # current tower context
    #     # calculate levels and sort
        
    #     levels = torch.matmul(level_codes, self.basis)
        
    #     levels, sort_id = torch.topk(torch.transpose(levels, 1, 0), num_levels)
    #     levels = torch.flip(levels, [-1])
    #     sort_id = torch.flip(sort_id, [-1])
    #     levels = torch.transpose(levels, 1, 0)
    #     sort_id = torch.transpose(sort_id, 1, 0)
    #     # calculate threshold
    #     thrs = torch.matmul(thrs_multiplier, levels)
    #     # calculate level codes per channel
    #     reshape_x = torch.reshape(x, [-1, num_filters])
        
    #     # level_codes_channelwise_dims = torch.stack([num_levels * num_filters, nbit])
    #     # level_codes_channelwise = level_codes_channelwise_dims.fill_(0.)# tf.fill(level_codes_channelwise_dims, 0.)
    #     level_codes_channelwise = torch.Tensor(num_levels * num_filters, nbit).fill_(0.)

    #     for i in range(num_levels):
    #         # eq = torch.equal(sort_id, i)
    #         import pdb; pdb.set_trace()
    #         eq = sort_id == i
    #         level_codes_channelwise = torch.where(torch.reshape(eq, [-1]), level_codes_channelwise + level_codes[i], level_codes_channelwise)
    #     level_codes_channelwise = torch.reshape(level_codes_channelwise, [num_levels, num_filters, nbit])
    #     import pdb; pdb.set_trace()
    #     # calculate output y and its binary code
    #     # NOTE: check if the function zeros_like work properly
    #     y = torch.zeros_like(x) + levels[0]  # output
    #     zero_dims = torch.stack([torch.shape(reshape_x)[0] * num_filters, nbit])
    #     bits_y = torch.fill(zero_dims, -1.)
    #     zero_y = torch.zeros_like(x)
    #     zero_bits_y = torch.fill(zero_dims, 0.)
    #     zero_bits_y = torch.reshape(zero_bits_y, [-1, num_filters, nbit])
    #     for i in range(num_levels - 1):
    #         g = x.gt(thrs[i])# tf.greater(x, thrs[i])
    #         y = torch.where(g, zero_y + levels[i + 1], y)
    #         bits_y = torch.where(torch.reshape(g, [-1]), torch.reshape(zero_bits_y + level_codes_channelwise[i + 1], [-1, nbit]), bits_y)
    #     bits_y = torch.reshape(bits_y, [-1, num_filters, nbit])
    #     # training
    #     # if ctx.is_main_training_tower:
    #     BT = torch.permute(bits_y, 2, 0, 1)
    #     # calculate BTxB
    #     BTxB = []
    #     for i in range(nbit):
    #         for j in range(nbit):
    #             BTxBij = torch.mul(BT[i], BT[j])
    #             BTxBij = torch.matmul(sum_multiplier, BTxBij)
    #             if i == j:
    #                 mat_one = torch.ones([1, num_filters])
    #                 BTxBij = BTxBij + (delta * mat_one)  # + E
    #             BTxB.append(BTxBij)
    #     BTxB = torch.reshape(torch.stack(values=BTxB), [nbit, nbit, num_filters])
    #     # calculate inverse of BTxB
    #     if nbit > 2:
    #         BTxB_transpose = torch.permute(BTxB, 2, 0, 1)
    #         BTxB_inv = torch.inverse(BTxB_transpose)
    #         BTxB_inv = torch.permute(BTxB_inv, 1, 2, 0)
    #     elif nbit == 2:
    #         det = torch.mul(BTxB[0][0], BTxB[1][1]) - torch.mul(BTxB[0][1], BTxB[1][0])
    #         inv = []
    #         inv.append(BTxB[1][1] / det)
    #         inv.append(-BTxB[0][1] / det)
    #         inv.append(-BTxB[1][0] / det)
    #         inv.append(BTxB[0][0] / det)
    #         BTxB_inv = torch.reshape(torch.stack(values=inv), [nbit, nbit, num_filters])
    #     elif nbit == 1:
    #         BTxB_inv = torch.reciprocal(BTxB)
    #     # calculate BTxX
    #     BTxX = []
    #     for i in range(nbit):
    #         BTxXi0 = torch.mul(BT[i], reshape_x)
    #         BTxXi0 = torch.matmul(sum_multiplier, BTxXi0)
    #         BTxX.append(BTxXi0)
    #     BTxX = torch.reshape(torch.stack(values=BTxX), [nbit, num_filters])
    #     BTxX = BTxX + (delta * self.basis)  # + basis
    #     # calculate new basis
    #     new_basis = []
    #     for i in range(nbit):
    #         new_basis_i = torch.mul(BTxB_inv[i], BTxX)
    #         new_basis_i = torch.matmul(sum_multiplier_basis, new_basis_i)
    #         new_basis.append(new_basis_i)
    #     new_basis = torch.reshape(torch.stack(values=new_basis), [nbit, num_filters])
    #     # create moving averages op
    #     # FIXME:how this moving average is used???
    #     # updata_moving_basis = moving_averages.assign_moving_average(
    #     #     basis, new_basis, MOVING_AVERAGES_FACTOR)
    #     # add_model_variable(basis)
    #     self.basis = self.basis.clone().detach() * (1-MOVING_AVERAGES_FACTOR) + new_basis * MOVING_AVERAGES_FACTOR
    #     # calculate moving of basis
    #     # tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, updata_moving_basis)

    #     y = x - x.clone().detach() + y .clone().detach()
    #     # y = x + tf.stop_gradient(-x) + tf.stop_gradient(y)  # gradient: y=x
    #     # y.variables = VariableHolder(basis=basis)
    #     return y
    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        # import pdb; pdb.set_trace()
        #oh, ow = torch.ceil(self.div(ih.float(), sh)), torch.ceil(self.div(iw.float(), sw))
        #pad_h = max(self.add(self.mul(self.add(oh.float(), -1), self.stride[0]) ,self.mul(self.add(kh, -1), self.add(self.add(self.dilation[0], 1), -ih))), 0)
        #pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        # import pdb; pdb.set_trace()
        # dims = self.weight.shape
        # ele_num = dims[0] * dims[1] * dims[2]
        # self.QuantizedWeight(self.weight, ele_num,self.nbits)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        out = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out
#class Identity(nn.Module):
#    def __init__(self,):
#        super(Identity, self).__init__()
#
#    def forward(self, input):
#        return input

class Conv2dSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""
    def __init__(self, in_channels, out_channels, kernel_size, image_size=224, **kwargs):
        super(Conv2dSamePadding, self).__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        #if pad_h > 0 or pad_w > 0:
        self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        #else:
            # self.static_padding = Identity()

    def forward(self, x):
        x_tmp = self.static_padding(x)
        # import pdb; pdb.set_trace()
        out = F.conv2d(x_tmp.half(), self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return out
class WSConv2d_v1_v0(nn.Conv2d):
    def __init__(self, in_planes, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, multiplier = 1.0, rep_dim = 1, repeat_weight = True, use_coeff=False, image_size = 224):
        if rep_dim == 0:
            # this is repeat along the channel dim (dimension 0 of the weights tensor)
            super(WSConv2d_v1, self).__init__(
                int(in_planes), int(np.ceil(out_channels/ multiplier)),
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias)
            self.rep_time = int(np.ceil(
            1. * out_channels / self.weight.shape[0]))
        elif rep_dim == 1:
            # this is to repeat along the filter dim(dimension 1 of the weights tensor)
            super(WSConv2d_v1, self).__init__(
                int(np.ceil(in_planes/ multiplier)),int(out_channels),
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias)
            self.rep_time = int(np.ceil(
            1. * in_planes / self.weight.shape[1]))

        self.in_planes = in_planes
        self.out_channels_ori = out_channels
        self.groups = groups
        self.multiplier = multiplier
        self.rep_dim = rep_dim
        self.repeat_weight = repeat_weight
        self.use_coeff = use_coeff
        self.coefficient = Parameter(torch.Tensor(self.rep_time), requires_grad=False)
        self.reuse = False
        self.coeff_grad = None

         # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        #if pad_h > 0 or pad_w > 0:
        self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    def forward(self, x):
        """
            same padding as efficientnet tf version
        """
        x = self.static_padding(x)
        if self.rep_dim == 0:
            #out = F.conv2d(x, self.weight.repeat([self.rep_time,1,1,1])[:self.out_channels_ori,:,:,:], None, 1)
            out = F.conv2d(x, self.weight.repeat([self.rep_time,1,1,1]), None, 1)
        else:
            #out = F.conv2d(x, self.weight.repeat([1,self.rep_time,1,1])[:,:x.shape[1],:,:], None, 1)
            out = F.conv2d(x, self.weight.repeat([1,self.rep_time,1,1]), None, 1)
        return out
class WSConv2d_v1(nn.Conv2d):
    def __init__(self, in_planes, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, multiplier = 1.0, rep_dim = 1, repeat_weight = True, use_coeff=False, image_size = 224):
        if rep_dim == 0:
            # this is repeat along the channel dim (dimension 0 of the weights tensor)
            super(WSConv2d_v1, self).__init__(
                int(in_planes), int(np.ceil(out_channels/ multiplier)),
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias)
            self.rep_time = int(np.ceil(
            1. * out_channels / self.weight.shape[0]))
        elif rep_dim == 1:
            # this is to repeat along the filter dim(dimension 1 of the weights tensor)
            super(WSConv2d_v1, self).__init__(
                int(np.ceil(in_planes/ multiplier)),int(out_channels), 
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias)
            self.rep_time = int(np.ceil(
            1. * in_planes / self.weight.shape[1]))

        self.in_planes = in_planes
        self.out_channels_ori = out_channels
        self.groups = groups
        self.multiplier = multiplier
        self.rep_dim = rep_dim
        self.repeat_weight = repeat_weight
        self.use_coeff = use_coeff
        self.coefficient = Parameter(torch.Tensor(self.rep_time), requires_grad=False)
        self.reuse = False
        self.coeff_grad = None

         # Calculate padding based on image size and save it
        #assert image_size is not None
        #ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        #kh, kw = self.weight.size()[-2:]
        #sh, sw = self.stride
        #oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        #pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        #pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        ##if pad_h > 0 or pad_w > 0:
        #self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
    def forward(self, x):
        """
            same padding as efficientnet tf version
        """
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        
        # print("use_coeff == Falsei")
        # x = self.static_padding(x)
        if self.rep_dim == 0:
            #out = F.conv2d(x, self.weight.repeat([self.rep_time,1,1,1])[:self.out_channels_ori,:,:,:], None, 1)
            out = F.conv2d(x, self.weight.repeat([self.rep_time,1,1,1]), None, 1)
        else:
            #out = F.conv2d(x, self.weight.repeat([1,self.rep_time,1,1])[:,:x.shape[1],:,:], None, 1)
            out = F.conv2d(x, self.weight.repeat([1,self.rep_time,1,1]), None, 1)
        return out
class WSConv2d(nn.Conv2d):
    def __init__(self, in_planes, out_channels,
                 kernel_size, stride=1, padding=0, dilation=1,
                 groups=1, bias=True, multiplier = 1.0, rep_dim = 1, repeat_weight = True, use_coeff=False):
        if rep_dim == 0:
            # this is repeat along the channel dim (dimension 0 of the weights tensor)
            super(WSConv2d, self).__init__(
                int(in_planes), int(np.ceil(out_channels/ multiplier)),
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias)
            self.rep_time = int(np.ceil(
            1. * out_channels / self.weight.shape[0]))
        elif rep_dim == 1:
            # this is to repeat along the filter dim(dimension 1 of the weights tensor)
            super(WSConv2d, self).__init__(
                int(np.ceil(in_planes/ multiplier)),int(out_channels), 
                kernel_size, stride=stride, padding=padding, dilation=dilation,
                groups=groups, bias=bias)
            self.rep_time = int(np.ceil(
            1. * in_planes / self.weight.shape[1]))

        self.in_planes = in_planes
        self.out_channels_ori = out_channels
        self.groups = groups
        self.multiplier = multiplier
        self.rep_dim = rep_dim
        self.repeat_weight = repeat_weight
        self.use_coeff = use_coeff
        # print(self.weight.shape)
        # import pdb; pdb.set_trace() 

        self.conv1_stride_lr_1 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=2, padding=0, bias=False)
        self.conv1_stride_lr_2 = nn.Conv2d(in_planes, self.rep_time, kernel_size=1, stride=1, padding=0, bias=False)
        self.coefficient = Parameter(torch.Tensor(self.rep_time), requires_grad=False)
        self.reuse = False
        self.coeff_grad = None

    def generate_share_weight(self, base_weight, rep_num, coeff, nchannel, dim = 0):
        ''' sample weights from base weight'''
        # pdb.set_trace()
        if rep_num == 1:
            return base_weight
        new_weight = []
        for i in range(rep_num):
            if dim == 0:
                new_weight_temp = torch.cat([base_weight[1:,:,:,:],
                    base_weight[0:1,:,:,:]], dim=0) * (1 - coeff[i])
            else:
                new_weight_temp = torch.cat([base_weight[:,1:,:,:],
                    base_weight[:,0:1,:,:]], dim=1) * (1 - coeff[i])
            new_weight.append(base_weight * coeff[i] + new_weight_temp)
        out = torch.cat(new_weight, dim=dim)
        
        if dim == 0:
            return out[:nchannel,:,:,:]
        else:
            return out[:,:nchannel,:,:]

    def forward(self, x):
        """
            same padding as efficientnet tf version
        """
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w//2, pad_w - pad_w//2, pad_h//2, pad_h - pad_h//2])
        
        if self.use_coeff:
            if self.training:
                # set reuse to True for coefficient sharing
                if not self.reuse:
                    lr_conv1 = self.conv1_stride_lr_1(x)
                    # pdb.set_trace()
                    lr_conv1 = self.conv1_stride_lr_2(lr_conv1)
                    lr_conv1 = F.adaptive_avg_pool2d(lr_conv1, (1,1))[:,:,0,0]

                    self.coefficient.set_(F.normalize(torch.mean(lr_conv1, 0), dim = 0).clone().detach())
                    # pdb.set_trace()
                    self.coeff_grad = F.normalize(torch.mean(lr_conv1, 0), dim = 0)

                if self.rep_dim == 0:
                    if self.repeat_weight:
                        out = F.conv2d(x, self.generate_share_weight(
                        self.weight, self.rep_time,
                        self.coeff_grad, self.out_channels_ori, dim = 0))
                    else:
                        out_tmp = F.conv2d(x, self.weight)
                        out = self.generate_share_feature(self.rep_time,out_tmp, F.normalize(torch.mean(lr_conv1, 0), dim = 0), self.out_channels_ori)
                        # out = F.conv2d(x, self.generate_share_feature(self.rep_time, F.normalize(torch.mean(lr_conv1, 0), dim = 0)))
                else:
                    if self.repeat_weight:
                        
                        out = F.conv2d(x, self.generate_share_weight(
                        self.weight, self.rep_time,
                        self.coeff_grad, x.shape[1], dim=1))
                    else:
                        out_tmp = self.feature_wrapper(self.rep_time,x, F.normalize(torch.mean(lr_conv1, 0), dim = 0), self.out_channels_ori)
                        out = F.conv2d(out_tmp, self.weight)

            else:
                if self.rep_dim == 0:
                    if self.repeat_weight:
                        out = F.conv2d(x, self.generate_share_weight(
                        self.weight, self.rep_time,
                        self.coefficient.detach(), self.out_channels_ori, dim = 0))
                    else:
                        out_tmp = F.conv2d(x, self.weight)
                        out = self.generate_share_feature(self.rep_time,out_tmp, self.coefficient.detach(), self.out_channels_ori)
                        # out = F.conv2d(x, self.generate_share_feature(self.rep_time, self.coefficient.detach()))
                else:
                    if self.repeat_weight:
                        out = F.conv2d(x, self.generate_share_weight(
                        self.weight, self.rep_time,
                        self.coefficient.detach(), x.shape[1], dim = 1))
                    else:
                        out_tmp = self.feature_wrapper(self.rep_time,out_tmp, self.coefficient.detach(), self.out_channels_ori)
                        
                        out = F.conv2d(out_tmp, self.weight)
        else:
            # print("use_coeff == False")
            if self.rep_dim == 0:
                out = F.conv2d(x, self.weight.repeat([self.rep_time,1,1,1])[:self.out_channels_ori,:,:,:], None, 1)
            else:
                out = F.conv2d(x, self.weight.repeat([1,self.rep_time,1,1])[:,:x.shape[1],:,:], None, 1)
        return out



########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None,
                 dropout_rate=0.2, drop_connect_rate=0.2):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=1000,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, _, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(width_coefficient=w, depth_coefficient=d, dropout_rate=p)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


url_map = {
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth',
}

def load_pretrained_weights(model, model_name):
    """ Loads pretrained weights, and downloads if loading for the first time. """
    state_dict = model_zoo.load_url(url_map[model_name])
    model.load_state_dict(state_dict)
    print('Loaded pretrained weights for {}'.format(model_name))
