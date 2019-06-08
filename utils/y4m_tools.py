import gc
import os
import glob
import time
import cv2
import numpy as np
from collections import Counter
from data.youku import YoukuDataset


def read_y4m(file_path, mode="444"):
    """
    读取y4m视频文件为4维ndarray
    :param file_path: the path of a y4m file (420planar).
    :param mode: 元素排列格式
    :return: a ndarray with the shape of (nFrames, height, width, 3) and meta data
    """
    with open(file_path, 'rb') as fp:
        header = fp.readline()
        raw_frames = fp.read().split(b'FRAME\n')[1:]

    width = int(header.split()[1][1:])
    height = int(header.split()[2][1:])
    uvw, uvh = width >> 1, height >> 1
    n_pixel = height * width
    yuv_frames = [np.frombuffer(frame, dtype=np.uint8) for frame in raw_frames]
    start_y, end_y = (0, n_pixel)
    start_u, end_u = (n_pixel, n_pixel + (n_pixel >> 2))
    start_v, end_v = (n_pixel + (n_pixel >> 2), n_pixel + (n_pixel >> 1))
    frames = list()

    def extend_uv(c: np.ndarray) -> np.ndarray:
        cc = c.repeat(2)
        cc_cc = np.vstack((cc, cc))
        c4 = np.hsplit(cc_cc, c.shape[0])
        return np.vstack(c4)

    def make_frame(yc, uc, vc):
        if mode == "444":
            uc, vc = extend_uv(uc), extend_uv(vc)
            return np.array([yc, uc, vc]).transpose((1, 2, 0))
        return

    for frame in yuv_frames:
        y = np.array((frame[:end_y]), dtype=np.uint8).reshape(height, width)
        u = np.array(frame[start_u:end_u], dtype=np.uint8).reshape(uvh, uvw)
        v = np.array(frame[start_v:end_v], dtype=np.uint8).reshape(uvh, uvw)
        frames.append(make_frame(y, u, v))

    del raw_frames, yuv_frames
    gc.collect()
    return np.array(frames), header


def convert(data_dir):
    """
    对文件夹目录下所有y4m文件分帧按文件夹存放。
    :param data_dir: data dir
    """
    nsh, nsl = np.array([1088, 1920]), np.array([272, 480])
    t0 = time.time()
    path_list = sorted(glob.glob(f"{data_dir}/*.y4m"))
    print("There are", len(path_list), "y4m videos in ", data_dir)
    for i, v_path in enumerate(path_list, 1):
        v_name = os.path.basename(v_path)[:-4]
        print("\r" + f"Processing {v_name}.  {i / len(path_list):.2%}", end="")
        frames, header = read_y4m(v_path)
        shape = np.array(frames.shape[1:3])
        if b'W2048' in header:
            p = (shape - nsh) >> 1
            q = p + nsh
            header.replace(b'W2048', b'W1920')
            header.replace(b'H1152', b'H1088')
            frames = frames[:, p[0]:q[0], p[1]:q[1], :]
        elif b'W512' in header:
            p = (shape - nsl) >> 1
            q = p + nsl
            header.replace(b'W512', b'W480')
            header.replace(b'H288', b'H272')
            frames = frames[:, p[0]:q[0], p[1]:q[1], :]
        elif b'W1920' in header:
            header.replace(b'W2048', b'W1920')
            header.replace(b'H1152', b'H1088')
            frames = np.pad(frames, ((0, 0), (8, 0), (0, 0), (0, 0)),
                            'constant', constant_values=(0, 0))
        elif b'W480' in header:
            header.replace(b'W512', b'W480')
            header.replace(b'H288', b'H272')
            frames = np.pad(frames, ((0, 0), (2, 0), (0, 0), (0, 0)),
                            'constant', constant_values=(0, 0))
        # print(f'{v_name}   {frames.shape} \n')
        fid_len = len(str(len(frames) - 1))
        im_dir = v_path[:-4]  # 分帧存放文件夹
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        fid_len = len(str(len(frames) - 1))
        # save frames
        for n, f in enumerate(frames):
            file_name = f"{v_name}_{len(frames)}_{str(n).zfill(fid_len)}_.npy"
            img_path = f"{im_dir}/{file_name}"
            np.save(img_path, f)
    else:
        t1 = time.time()
        print(f"\rSuccessful converted {len(path_list)} videos in {t1 - t0:.4f} sec.", end="")
    return


def save_y4m(yuv420p_imgs, header, save_path):
    """
    :param yuv420p_imgs: yuv420p 格式图片s
    :param header: 该视频的头
    :param save_path: 存储路径
    :return:
    """
    with open(save_path, 'wb') as v:
        v.write(header)
        for frame in yuv420p_imgs:
            v.write(b'FRAME\n')
            v.write(frame.tostring())
    return


def resize(image, width=None, height=None, inter=cv2.INTER_LINEAR):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    if height is None:
        r = width / float(w)
        dim = (width, int(h * r))

    if width and height:
        dim = (width, height)

    resized = cv2.resize(image, dim, interpolation=inter)
    return resized


def yuv444to420p(img: np.ndarray, inter=cv2.INTER_LINEAR) -> np.ndarray:
    """
    yuv444转420p，可自定义方法
    :param img: 图像
    :param inter: 插值方法
    :return: 一维数组
    """
    [y, u, v] = img.astype(np.uint8).transpose((2, 0, 1))
    w, h = y.shape[1] >> 1, y.shape[0] >> 1
    u, v = resize(u, w, h, inter), resize(v, w, h, inter=inter)
    return np.concatenate([y.reshape(-1), u.reshape(-1), v.reshape(-1)])


def test():
    c = list()
    cao = list()
    path_list = glob.glob(f"D:\\DATA\\train\\*.y4m")
    for file_path in path_list:
        with open(file_path, 'rb') as fp:
            header = fp.readline().split()
            if header[1] == b"W512" or header[1] == b"W2048":
                cao.append(os.path.basename(file_path))
            c.append(header[1] + header[2])
    c = Counter(c)

    return


if __name__ == '__main__':
    DIR = r"D:\DATA\train"
    # imgs, _ = read_y4m("../dataset/train/Youku_00000_l.y4m")
    # fs = [yuv444to420p(i) for i in imgs]
    # save_y4m(fs, "../dataset/train/Youku_00000_l/header.txt", "../results/Youku_00000_l.y4m")
    # convert(DIR)
    yk = YoukuDataset(DIR, 4, 7, False, 64, "new_info", v_freq=5, cut=True)
    yk.__getitem__(0)
    # test()
    pass
    # header: 'signature width height fps interlacing pixelAspectRadio colorSpace comment'
