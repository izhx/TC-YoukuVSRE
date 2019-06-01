import gc
import os
import glob
import time
import cv2
import numpy as np
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

    def make_frame(y, u, v):
        if mode == "444":
            u, v = extend_uv(u), extend_uv(v)
            return np.array([y, u, v]).transpose((1, 2, 0))
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
    t0 = time.time()
    path_list = glob.glob(f"{data_dir}/*.y4m")
    print("There are", len(path_list), "y4m videos in ", data_dir)
    for i, v_path in enumerate(path_list, 1):
        v_name = os.path.basename(v_path)[:-4]
        print("\r" + f"Processing {v_name}.  {i / len(path_list):.2%}", end="")
        frames, header = read_y4m(v_path)
        im_dir = v_path[:-4]  # 分帧存放文件夹
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        fid_len = len(str(len(frames) - 1))
        # save frames and header
        for n, f in enumerate(frames):
            file_name = f"{v_name}_{len(frames)}_{str(n).zfill(fid_len)}_.npy"
            img_path = f"{im_dir}/{file_name}"
            np.save(img_path, f)
        with open(os.path.join(im_dir, "header.txt"), 'wb') as f:
            f.write(header)
    else:
        t1 = time.time()
        print(f"\rSuccessful converted {len(path_list)} videos in {t1 - t0:.4f} sec.", end="")
    return


def save_y4m(yuv420p_imgs, header_path, save_path):
    """
    :param yuv420p_imgs: yuv420p 格式图片s
    :param header_path: 该视频的头
    :param save_path: 存储路径
    :return:
    """
    with open(header_path, 'rb') as h:
        header = h.readline()
    with open(save_path, 'wb') as v:
        v.write(header)
        for frame in yuv420p_imgs:
            v.write(b'FRAME\n')
            v.write(frame.tobytes())
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


if __name__ == '__main__':
    # yk = YoukuDataset("../dataset/train", 4, 5, True, 31, "new_info")
    DIR = "../dataset/train"
    imgs, _ = read_y4m("../dataset/train/Youku_00000_l.y4m")
    fs = [yuv444to420p(i) for i in imgs]
    #save_y4m(fs, "../dataset/train/Youku_00000_l/header.txt", "../results/Youku_00000_l.y4m")
    convert(DIR)

    # header: 'signature width height fps interlacing pixelAspectRadio colorSpace comment'