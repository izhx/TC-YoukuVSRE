import torch.nn as nn
import torchvision

_VIDEO_KIND = {0: 0, 1: 1, 2: 2}


class Youku(nn.Module):
    def __init__(self, scale_factor, video_kind=2):
        super(Youku, self).__init__()
        self.scale_factor = scale_factor
        self.classifier = torchvision.models.resnet34(num_classes=video_kind, zero_init_residual=False)
        self.SR_Real = None
        self.SR_Carton = None
