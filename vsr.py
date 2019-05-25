from VSR.RBPN.youku import YoukuDataset

import argparse
from collections import OrderedDict
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from VSR.RBPN.rbpn import RBPN
from VSR.RBPN.youku import get_eval_set
import numpy as np
import time
import cv2
import math

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=4, help="super resolution upscale factor")
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--gpu_mode', type=bool, default=True)
parser.add_argument('--chop_forward', type=bool, default=False)
parser.add_argument('--threads', type=int, default=0, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--gpus', default=1, type=int, help='number of gpu')
parser.add_argument('--data_dir', type=str, default='./dataset')
parser.add_argument('--file_list', type=str, default='foliage.txt')
parser.add_argument('--other_dataset', type=bool, default=True, help="use other dataset than vimeo-90k")
parser.add_argument('--future_frame', type=bool, default=True, help="use future frame")
parser.add_argument('--nFrames', type=int, default=7)
parser.add_argument('--model_type', type=str, default='RBPN')
parser.add_argument('--residual', type=bool, default=False)
parser.add_argument('--output', default='./VSR/RBPN/Results/', help='Location to save checkpoint models')
parser.add_argument('--model', default='./VSR/RBPN/weights/RBPN_4x.pth', help='sr pretrained base model')

opt = parser.parse_args()

gpu_list = range(opt.gpus)
print(opt)

_PREFIX = 'dataset/train/Youku_00000'

cuda = opt.gpu_mode
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

print('===> Loading dataset')
test_set = get_eval_set(_PREFIX, opt.upscale_factor, False, 0, opt.future_frame)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False)

print('===> Building model ', opt.model_type)
if opt.model_type == 'RBPN':
    model = RBPN(num_channels=3, base_filter=256, feat=64, num_stages=3, n_resblock=5, n_frames=opt.nFrames,
                 scale_factor=opt.upscale_factor)
else:
    model = None

if cuda:
    model = torch.nn.DataParallel(model, device_ids=gpu_list)

if cuda:
    model.load_state_dict(torch.load(opt.model, map_location=lambda storage, loc: storage))
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

print('Pre-trained SR model is loaded.')

if cuda:
    model = model.cuda(gpu_list[0])


def eval_fun():
    model.eval()
    count = 1
    # avg_psnr_predicted = 0.0
    for batch in testing_data_loader:
        lr, target, neighbor, flow, bicubic = batch[0], batch[1], batch[2], batch[3], batch[4]

        with torch.no_grad():
            if cuda:
                lr = Variable(lr).cuda(gpu_list[0])
                bicubic = Variable(bicubic).cuda(gpu_list[0])
                neighbor = [Variable(j).cuda(gpu_list[0]) for j in neighbor]
                flow = [Variable(j).cuda(gpu_list[0]).float() for j in flow]
            else:
                lr = Variable(lr)
                bicubic = Variable(bicubic)
                neighbor = [Variable(j) for j in neighbor]
                flow = [Variable(j).float() for j in flow]

        t0 = time.time()
        if opt.chop_forward:
            with torch.no_grad():
                prediction = chop_forward(lr, neighbor, flow, model, opt.upscale_factor)
        else:
            with torch.no_grad():
                prediction = model(lr, neighbor, flow)

        if opt.residual:
            prediction = prediction + bicubic

        t1 = time.time()
        print("===> Processing: %s || Timer: %.4f sec." % (str(count), (t1 - t0)))
        save_img(prediction.cpu().data, str(count), True)
        save_img(target, str(count), False)

        # prediction=prediction.cpu()
        # prediction = prediction.data[0].numpy().astype(np.float32)
        # prediction = prediction*255.

        # target = target.squeeze().numpy().astype(np.float32)
        # target = target*255.

        # psnr_predicted = PSNR(prediction,target, shave_border=opt.upscale_factor)
        # avg_psnr_predicted += psnr_predicted
        count += 1

    # print("PSNR_predicted=", avg_psnr_predicted/count)


def save_img(img, img_name, pred_flag):
    saving_img = img.squeeze().clamp(0, 1).numpy().transpose(1, 2, 0)

    # save img
    save_dir = os.path.join(opt.output, opt.data_dir,
                            os.path.splitext(opt.file_list)[0] + '_' + str(opt.upscale_factor) + 'x')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if pred_flag:
        save_fn = save_dir + '/' + img_name + '_' + opt.model_type + 'F' + str(opt.nFrames) + '.png'
    else:
        save_fn = save_dir + '/' + img_name + '.png'
    cv2.imwrite(save_fn, cv2.cvtColor(saving_img * 255, cv2.COLOR_BGR2RGB), [cv2.IMWRITE_PNG_COMPRESSION, 0])


def psnr(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[1 + shave_border:height - shave_border, 1 + shave_border:width - shave_border, :]
    gt = gt[1 + shave_border:height - shave_border, 1 + shave_border:width - shave_border, :]
    im_dff = pred - gt
    r_mse = math.sqrt(np.mean(im_dff ** 2))
    if r_mse == 0:
        return 100
    return 20 * math.log10(255.0 / r_mse)


def chop_forward(x, neighbor, flow, the_model, scale, shave=8, min_size=2000, nGPUs=opt.gpus):
    b, c, h, w = x.size()
    h_half, w_half = h // 2, w // 2
    h_size, w_size = h_half + shave, w_half + shave
    input_list = [
        [x[:, :, 0:h_size, 0:w_size], [j[:, :, 0:h_size, 0:w_size] for j in neighbor],
         [j[:, :, 0:h_size, 0:w_size] for j in flow]],
        [x[:, :, 0:h_size, (w - w_size):w], [j[:, :, 0:h_size, (w - w_size):w] for j in neighbor],
         [j[:, :, 0:h_size, (w - w_size):w] for j in flow]],
        [x[:, :, (h - h_size):h, 0:w_size], [j[:, :, (h - h_size):h, 0:w_size] for j in neighbor],
         [j[:, :, (h - h_size):h, 0:w_size] for j in flow]],
        [x[:, :, (h - h_size):h, (w - w_size):w], [j[:, :, (h - h_size):h, (w - w_size):w] for j in neighbor],
         [j[:, :, (h - h_size):h, (w - w_size):w] for j in flow]]]

    if w_size * h_size < min_size:
        output_list = []
        for i in range(0, 4, nGPUs):
            with torch.no_grad():
                input_batch = input_list[i]  # torch.cat(input_list[i:(i + nGPUs)], dim=0)
                output_batch = the_model(input_batch[0], input_batch[1], input_batch[2])
            output_list.extend(output_batch.chunk(nGPUs, dim=0))
    else:
        output_list = [
            chop_forward(patch[0], patch[1], patch[2], the_model, scale, shave, min_size, nGPUs)
            for patch in input_list]

    h, w = scale * h, scale * w
    h_half, w_half = scale * h_half, scale * w_half
    h_size, w_size = scale * h_size, scale * w_size
    shave *= scale

    with torch.no_grad():
        output = Variable(x.data.new(b, c, h, w))
    output[:, :, 0:h_half, 0:w_half] \
        = output_list[0][:, :, 0:h_half, 0:w_half]
    output[:, :, 0:h_half, w_half:w] \
        = output_list[1][:, :, 0:h_half, (w_size - w + w_half):w_size]
    output[:, :, h_half:h, 0:w_half] \
        = output_list[2][:, :, (h_size - h + h_half):h_size, 0:w_half]
    output[:, :, h_half:h, w_half:w] \
        = output_list[3][:, :, (h_size - h + h_half):h_size, (w_size - w + w_half):w_size]

    return output


def main():  # for test
    ykd = YoukuDataset('dataset/Youku_00000', 4, True, 0, True)
    ykd.__getitem__(0)
    return


# Eval Start!!!!
if __name__ == '__main__':
# main()
    eval_fun()