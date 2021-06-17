## ECCV-2018-Image Super-Resolution Using Very Deep Residual Channel Attention Networks
## https://arxiv.org/abs/1807.02758
from model import common

import torch.nn as nn
import torch

def make_model(args, parent=False):
    return RCAN(args)

def get_valid_padding(kernel_size, dilation):
    kernel_size = kernel_size + (kernel_size - 1) * (dilation - 1)
    padding = (kernel_size - 1) // 2
    return padding

class AdaptiveFM(nn.Module):
    # hello ckx
    def __init__(self, in_channel, kernel_size):

        super(AdaptiveFM, self).__init__()
        padding = get_valid_padding(kernel_size, 1)
        self.transformer = nn.Conv2d(in_channel, in_channel, kernel_size,
                                     padding=padding, groups=in_channel)
        self.gamma = nn.Parameter(torch.zeros(1), requires_grad=True)
    def forward(self, x):
        return self.transformer(x) * self.gamma + x

## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y

## Residual Channel Attention Block (RCAB)
class RCAB(nn.Module):
    def __init__(
        self, args, conv, n_feat, kernel_size, reduction,
        bias=True, bn=False, act=nn.ReLU(True), res_scale=1,segnum=1):

        super(RCAB, self).__init__()
        # modules_body = []
        # for i in range(2):
        #     modules_body.append(conv(n_feat, n_feat, kernel_size, bias=bias))
        #     if bn: modules_body.append(nn.BatchNorm2d(n_feat))
        #     if i == 0: modules_body.append(act)

        self.modules_body = common.ResBlock_rcan(conv, n_feat, kernel_size, args, bn=False, act=act, res_scale=args.res_scale)
        self.calayer = CALayer(n_feat, reduction)
        #modules_body.append(CALayer(n_feat, reduction))
        # if args.adafm:
            #self.body = nn.ModuleList(modules_body)
        # else:
            #self.body = nn.Sequential(*modules_body)
        # self.res_scale = res_scale

    def forward(self, x,num):
        res = self.modules_body(x,num)
        res = self.calayer(res)
        # res = self.body(x)
        #res = self.body(x).mul(self.res_scale)
        res += x
        return res

## Residual Group (RG)
class ResidualGroup(nn.Module):
    def __init__(self, args, conv, n_feat, kernel_size, reduction, act, res_scale, n_resblocks, segnum = 1):
        super(ResidualGroup, self).__init__()
        self.n_resblocks = n_resblocks
        # modules_body = []
        modules_body = [
            RCAB(
                args,conv, n_feat, kernel_size, reduction, bias=True, bn=False, act=nn.ReLU(True), res_scale=1, segnum = segnum) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(n_feat, n_feat, kernel_size))
        if args.adafm:
            self.body = nn.ModuleList(modules_body)
            #self.body = nn.Sequential(*modules_body)
        else:
            self.body = nn.Sequential(*modules_body)

    def forward(self, x,num):
        res = x
        for i in range(self.n_resblocks):
            res = self.body[i](res,num)
        res = self.body[-1](res)
        # res = self.body(x)
        res += x
        return res

## Residual Channel Attention Network (RCAN)
class RCAN(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(RCAN, self).__init__()
        
        n_resgroups = args.n_resgroups
        self.n_resgroups = n_resgroups
        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3
        reduction = args.reduction 
        scale = args.scale[0]
        self.args = args
        act = nn.ReLU(True)
        
        # RGB mean for DIV2K
        self.sub_mean = common.MeanShift(args.rgb_range)
        
        # define head module
        modules_head = [conv(args.n_colors, n_feats, kernel_size)]

        # define body module
        modules_body = [
            ResidualGroup(
                self.args,conv, n_feats, kernel_size, reduction, act=act, res_scale=args.res_scale, n_resblocks=n_resblocks, segnum=args.segnum) \
            for _ in range(n_resgroups)]

        modules_body.append(conv(n_feats, n_feats, kernel_size))

        # define tail module
        modules_tail = [
            common.Upsampler(conv, scale, n_feats,kernel_size, act=False),
            conv(n_feats, args.n_colors, kernel_size)]

        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        self.head = nn.Sequential(*modules_head)
        if args.adafm:
            self.body = nn.ModuleList(modules_body)
            #self.body = nn.Sequential(*modules_body)
        else:
            self.body = nn.Sequential(*modules_body)
        self.tail = nn.Sequential(*modules_tail)

    def forward(self, x, num):
        x = self.sub_mean(x)
        x = self.head(x)
        res = x
        for i in range(self.n_resgroups):
            res = self.body[i](res,num)
        res = self.body[-1](res)
        # res = self.body(x)
        res += x

        x = self.tail(res)
        x = self.add_mean(x)

        return x 

    def load_state_dict(self, state_dict, strict=False):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') >= 0:
                        print('Replace pre-trained upsampler to new one...')
                    else:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))

        if strict:
            missing = set(own_state.keys()) - set(state_dict.keys())
            if len(missing) > 0:
                raise KeyError('missing keys in state_dict: "{}"'.format(missing))
