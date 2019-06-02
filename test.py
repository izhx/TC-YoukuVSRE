import os
import time
import glob
import argparse
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader

from model.EDVR_arch import EDVR, CharbonnierLoss
from data.youku import YoukuDataset
from utils.y4m_tools import read_y4m, save_y4m

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./dataset/test', help="测试集路径")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=0, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=False)
parser.add_argument('--padding', type=str, default="reflection",
                    help="padding: replicate | reflection | new_info | circle")
parser.add_argument('--model_type', type=str, default='EDVR')
parser.add_argument('--pretrained_sr', default='weights/4x_edvr_epoch.pth', help='sr pretrained base model')
parser.add_argument('--pretrained', type=bool, default=True)
parser.add_argument('--result_dir', default='./result', help='Location to save result.')

opt = parser.parse_args()
gpus_list = range(opt.gpus)
cudnn.benchmark = True

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)
device = torch.device('cuda')

print(opt)

print('===> Loading dataset')
train_set = YoukuDataset(opt.data_dir, opt.upscale_factor, opt.nFrames,
                         opt.data_augmentation, opt.patch_size, opt.padding, v_freq=5)
data_loader = DataLoader(dataset=train_set, batch_size=opt.batchSize,
                         shuffle=True, num_workers=opt.threads,
                         collate_fn=train_set.collate_fn)

print('===> Building model ', opt.model_type)
if opt.model_type == 'EDVR':
    model = EDVR(64, opt.nFrames, groups=8, front_RBs=5, back_RBs=40)  # TODO edvr参数
else:
    model = None

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

criterion = CharbonnierLoss(reduce=True)

if opt.pretrained:
    model_name = os.path.join(opt.save_folder + opt.pretrained_sr)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
    criterion = criterion.cuda(gpus_list[0])
else:
    # original saved file with DataParallel
    state_dict = torch.load(opt.model, map_location=lambda storage, loc: storage)
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    # load params
    model.load_state_dict(new_state_dict)

optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.9, 0.999), eps=1e-8)


def single_forward(model, imgs_in):
    with torch.no_grad():
        model_output = model(imgs_in)
        if isinstance(model_output, list) or isinstance(model_output, tuple):
            output = model_output[0]
        else:
            output = model_output
    return output


def index_generation(crt_i, max_n, N, padding='reflection'):
    """
    padding: replicate | reflection | new_info | circle
    """
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:
            if padding == 'replicate':
                add_idx = 0
            elif padding == 'reflection':
                add_idx = -i
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)
            elif padding == 'circle':
                add_idx = N + i
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:
            add_idx = i
        return_l.append(add_idx)
    return return_l


def single_test(video_path):
    fac = opt.upscale_factor
    frames, header = read_y4m(video_path)
    header = header.split()
    frame_num = len(frames)
    # todo 处理 header

    avgpool = torch.nn.AvgPool2d((2, 2), stride=(2, 2))
    frames = np.stack(frames, axis=0)
    size = np.array(frames.shape)[1:3]
    hr_size = size * fac
    pad_size = (np.ceil(size / 4) * 4 - size).astype(np.int)
    frames = np.pad(frames, ((0, 0), (pad_size[0], pad_size[1]), (0, 0), (0, 0)), 'constant',
                    constant_values=(0, 0))
    imgs = torch.from_numpy(np.ascontiguousarray(frames.transpose((0, 3, 1, 2)))).float()

    def convert_channel(ch: torch.tensor):
        ch = ch.flatten().numpy()
        ch = (ch * 255).astype(np.uint8)
        return ch

    hr_frames = list()
    for i in range(frame_num):
        select_idx = index_generation(i, frame_num, opt.nFrames, padding=opt.padding)
        imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
        output = single_forward(model, imgs_in)
        output_f = output.data.float().cpu().squeeze(0)
        prediction_pool = avgpool(output_f[(1, 2), :, :])
        # 给出像素  todo 处理padding
        y = convert_channel(output_f[0, :, :])
        u = convert_channel(prediction_pool[0, :, :])
        v = convert_channel(prediction_pool[1, :, :])
        hr_frames.append(np.concatenate(y, u, v))

    header[1] = b'W' + bytes(hr_size[1])
    header[2] = b'H' + bytes(hr_size[0])
    save_path = f'{opt.result_dir}/{os.path.basename(video_path).replace("_l", "_h_Res")}'
    save_y4m(hr_frames, b' '.join(header), save_path)
    return


def test():
    test_paths = glob.glob(f"{opt.data_dir}/*_l.y4m")
    for vp in test_paths:
        single_test(vp)
    return


if __name__ == '__main__':
    test()
