import os
import time
import glob
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import cv2

from model.EDVR_arch import EDVR
from utils.y4m_tools import read_y4m, save_y4m
from data.info_list import SCENE

parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./dataset/train', help="测试集路径")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--patch_size', type=int, default=0, help='0 to use original frame size')
parser.add_argument('--data_augmentation', type=bool, default=False)
parser.add_argument('--padding', type=str, default="reflection",
                    help="padding: replicate | reflection | new_info | circle")
parser.add_argument('--model_type', type=str, default='EDVR')
parser.add_argument('--pretrained_sr', default='./weights/4x_EDVRyk_epoch_139.pth', help='sr pretrained base model')
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

print('===> Building model ', opt.model_type)
if opt.model_type == 'EDVR':
    model = EDVR(64, opt.nFrames, groups=8, front_RBs=5, back_RBs=40)  # TODO edvr参数
else:
    model = None

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpus_list)

if opt.pretrained:
    model_name = os.path.join(opt.pretrained_sr)
    if os.path.exists(model_name):
        model.load_state_dict(torch.load(model_name, map_location=lambda storage, loc: storage))
        print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpus_list[0])
else:
    # # original saved file with DataParallel
    # state_dict = torch.load(opt.model, map_location=lambda storage, loc: storage)
    # # create new OrderedDict that does not contain `module.`
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
    #     name = k[7:]  # remove `module.`
    #     new_state_dict[name] = v
    # # load params
    # model.load_state_dict(new_state_dict)
    pass


def save_img(yuv, name):
    yuv = np.transpose(yuv, (1, 2, 0))
    yuv = (yuv * 255.0).round().astype(np.uint8)
    img = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
    cv2.imwrite(f'./result/{name}.png', img)
    return


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
    if max_n < N:
        padding = 'replicate'

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


avgpool = torch.nn.AvgPool2d((2, 2), stride=(2, 2))


def single_test(video_path):
    fac = opt.upscale_factor
    print(f'Processing: {video_path}')
    t0 = time.time()
    frames, header = read_y4m(video_path)
    header = header.split()
    vid = os.path.basename(video_path)[:-6]
    # 转场切分
    scl = SCENE[vid]
    scenes = list()
    for i in range(1, len(scl)):
        scenes.append(frames[scl[i - 1]:scl[i], :, :, :])
    else:
        scenes.append(frames[scl[-1]:, :, :, :])

    size = np.array(frames[0].shape[:2])
    pad_size = (np.ceil(size / 4) * 4 - size).astype(np.int)
    hr_size, hr_pad = size * fac, pad_size * fac

    def convert_channel(ch: torch.tensor):
        ch = ch.numpy().flatten()
        ch = (ch * 255).round().astype(np.uint8)
        # Important. Unlike MATLAB, numpy.unit8() WILL NOT round by default.
        return ch

    hr_frames = list()
    for frames in scenes:
        # 归一化
        frames = frames.astype(np.float32)
        for i in range(len(frames)):
            img = frames[i]
            _min, _max = img.min(), img.max()
            frames[i] = (img - _min) / (_max - _min)
        # 预处理
        frames = np.stack(frames, axis=0)
        frames = np.pad(frames, ((0, 0), (pad_size[0], pad_size[1]), (0, 0), (0, 0)), 'constant',
                        constant_values=(0, 0))
        imgs = torch.from_numpy(np.ascontiguousarray(frames.transpose((0, 3, 1, 2)))).float()
        lfs = len(frames)
        # 单帧超分
        for i in range(lfs):
            select_idx = index_generation(i, lfs, opt.nFrames, padding=opt.padding)
            imgs_in = imgs.index_select(0, torch.LongTensor(select_idx)).unsqueeze(0).to(device)
            output = single_forward(model, imgs_in)
            output_f = output.data.float().cpu().squeeze(0)
            output_f = output_f[:, hr_pad[0]:, hr_pad[1]:]
            prediction_pool = avgpool(output_f)
            # 给出像素
            y = convert_channel(output_f[0, :, :])
            u = convert_channel(prediction_pool[1, :, :])
            v = convert_channel(prediction_pool[2, :, :])
            hr_frames.append(np.concatenate((y, u, v)))

    header[1] = b'W' + str(hr_size[1]).encode()
    header[2] = b'H' + str(hr_size[0]).encode()
    save_path = f'{opt.result_dir}/{os.path.basename(video_path).replace("_l", "_h_Res")}'
    header = b' '.join(header) + b'\n'

    # 后9/10抽帧存储
    if int(vid[6:]) > 204:
        thin_frames = list()
        for i, f in enumerate(hr_frames):
            if i % 25 == 0:
                thin_frames.append(f)
        save_y4m(thin_frames, header, save_path)
    else:  # 存完整的
        save_y4m(hr_frames, header, save_path)
    t1 = time.time()
    print(f'One video saved: {save_path}, timer: {(t1 - t0):.4f} sec.')
    return


def test():
    test_paths = glob.glob(f"{opt.data_dir}/*_l.y4m")
    for vp in test_paths:
        single_test(vp)
    return


if __name__ == '__main__':
    test()
