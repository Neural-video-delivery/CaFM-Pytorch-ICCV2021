from model import common

import torch.nn as nn
import torch.nn.init as init

#url = {
 ##   'r20f64': ''
#}

def make_model(args, parent=False):
    return VDSRR(args)

class VDSRR(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(VDSRR, self).__init__()

        n_resblocks = args.n_resblocks
        n_feats = args.n_feats
        kernel_size = 3 
        self.numbers = n_resblocks - 2
        self.scale = int(args.scale[0])

        self.upsample = nn.Upsample(scale_factor=self.scale, mode='bicubic', align_corners=False)
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)

        def basic_block(args, in_channels, out_channels, act):
            return common.BasicBlock_(
                args, conv, in_channels, out_channels, kernel_size,
                bias=True, bn=False, act=act
            )

        # define body module
        m_head = []
        m_body = []
        m_tail = []
        m_head.append(conv(args.n_colors, n_feats, kernel_size, bias=True))
        m_head.append(nn.ReLU(True))

        for _ in range(n_resblocks - 2):
            m_body.append(basic_block(args, n_feats, n_feats, nn.ReLU(True)))

        m_tail.append(conv(n_feats, args.n_colors, kernel_size, bias=True))
        self.head = nn.Sequential(*m_head)
        if args.cafm:
            self.body = nn.ModuleList(m_body)
        else:
            self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x, num):
        x = self.upsample(x)
        x = self.sub_mean(x)

        #cafm
        res = x
        res = self.head(res)
        for i in range(self.numbers):
            res = self.body[i](res, num)
        res = self.tail(res)
        res += x

        x = self.add_mean(res)

        return x 

