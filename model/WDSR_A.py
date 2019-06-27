import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable


class Block(nn.Module):
    def __init__(self, n_feats, kernel_size, block_feats, wn, act=nn.ReLU(True),
                 res_scale=1):
        super(Block, self).__init__()
        self.res_scale = res_scale
        body = list()
        body.append(
            wn(nn.Conv2d(n_feats, block_feats, kernel_size, padding=kernel_size // 2)))
        body.append(act)
        body.append(
            wn(nn.Conv2d(block_feats, n_feats, kernel_size, padding=kernel_size // 2)))

        self.body = nn.Sequential(*body)

    def forward(self, x):
        res = self.body(x) * self.res_scale
        res += x
        return res


class MODEL(nn.Module):
    def __init__(self, cuda=True, scale=4, n_res=8, n_feats=64, block_feats=8,
                 res_scale=1, n_colors=3, kernel_size=3,
                 mean=(99.00332925, 124.7647323, 128.69159715)):
        super(MODEL, self).__init__()
        # hyper-params
        act = nn.ReLU(True)

        def wn(x):
            return torch.nn.utils.weight_norm(x)

        self.mean = torch.FloatTensor(mean).view([1, n_colors, 1, 1])
        if cuda:
            self.mean = self.mean.cuda()

        # define head module
        head = list()
        head.append(wn(nn.Conv2d(n_colors, n_feats, 3, padding=3 // 2)))

        # define body module
        body = list()
        for i in range(n_res):
            body.append(Block(n_feats, kernel_size, block_feats, wn, act, res_scale))

        # define tail module
        tail = list()
        out_feats = scale * scale * n_colors
        tail.append(wn(nn.Conv2d(n_feats, out_feats, 3, padding=3 // 2)))
        tail.append(nn.PixelShuffle(scale))

        # define skip module
        skip = list()
        skip.append(wn(nn.Conv2d(n_colors, out_feats, 5, padding=5 // 2)))
        skip.append(nn.PixelShuffle(scale))

        # make object members
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)
        self.skip = nn.Sequential(*skip)

        # 初始化权重，参考EDVR
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight, gain=init.calculate_gain('relu'))
                if m.bias is not None:
                    init.normal_(m.bias, 0.0001)
            elif isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=init.calculate_gain('relu'))
                if m.bias is not None:
                    init.normal_(m.bias, 0.0001)
        return

    def forward(self, x):
        x = (x - self.mean) / 127.5
        s = self.skip(x)
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        x += s
        x = x * 127.5 + self.mean
        return x
