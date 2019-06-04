import torch.nn

_ACTIVATION = {
    'ReLU': torch.nn.ReLU(True),
    'PReLU': torch.nn.PReLU(),
    'LeakyReLU': torch.nn.LeakyReLU(0.2, True),
    'Tanh': torch.nn.Tanh(),
    'Sigmoid': torch.nn.Sigmoid(),
    None: None
}

_BN = {
    'batch': lambda out_channels: torch.nn.BatchNorm2d(out_channels),
    'instance': lambda out_channels: torch.nn.InstanceNorm2d(out_channels),
    None: None
}


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=True, activation='PReLU',
                 norm=None):
        super(ConvBlock, self).__init__()
        self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size,
                                    stride, padding, dilation, groups, bias=bias)
        self.norm = _BN[norm](out_channels)
        self.activation = _ACTIVATION[activation]

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2,
                 padding=1, output_padding=0, dilation=1, groups=1, bias=True,
                 activation='PReLU', norm=None):
        super(DeconvBlock, self).__init__()
        self.deconv = torch.nn.ConvTranspose2d(in_channels, out_channels,
                                               kernel_size, stride, padding,
                                               output_padding, groups, bias,
                                               dilation)
        self.norm = _BN[norm](out_channels)
        self.activation = _ACTIVATION[activation]

    def forward(self, x):
        if self.norm is not None:
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out
