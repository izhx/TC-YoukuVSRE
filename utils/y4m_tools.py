import gc
import os
import time
import numpy as np


# from PIL import Image, ImageOps


def read_y4m(file_path):
    """
    读取y4m视频文件为4维ndarray

    :param file_path: the path of a y4m file (420planar).

    :return: a ndarray with the shape of (nFrames, height, width, 3) and meta data
    """
    with open(file_path, 'rb') as fp:
        header = fp.readline()
        raw_frames = fp.read().split(b'FRAME\n')[1:]

    meta_data = dict(zip('signature width height fps interlacing pixelAspectRadio colorSpace comment'.split(),
                         header.decode('ASCII').split()))

    width = int(meta_data['width'][1:])
    height = int(meta_data['height'][1:])
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

    for frame in yuv_frames:
        y = np.array((frame[:end_y]), dtype=np.uint8).reshape(height, width)
        u = np.array(frame[start_u:end_u], dtype=np.uint8).reshape(uvh, uvw)
        v = np.array(frame[start_v:end_v], dtype=np.uint8).reshape(uvh, uvw)
        # u = np.array(ImageOps.scale(Image.fromarray(u), 2, Image.BICUBIC))
        # v = np.array(ImageOps.scale(Image.fromarray(v), 2, Image.BICUBIC))
        u, v = extend_uv(u), extend_uv(v)
        frames.append(np.array([y, u, v]).transpose((1, 2, 0)))

    del header, raw_frames, yuv_frames
    gc.collect()
    return np.array(frames), meta_data


def convert(data_dir):
    """
    对文件夹目录下所有y4m文件分帧按文件夹存放。
    每个视频文件夹内有file list，总目录下也记录了总的。

    :param data_dir: data dir
    """
    t0 = time.time()
    open(os.path.join(data_dir, "all_file_list.txt"), 'w').close()
    videos = [v for v in os.listdir(data_dir) if v.endswith('y4m')]
    print("There are", len(videos), "y4m videos in ", data_dir)
    for i, v in enumerate(videos, 1):
        vid = v[6:13]
        print("\rProcessing " + f"{vid}.  {i / len(videos):.2%}", end="")
        v_path = f"{data_dir}/{v}"
        frames, meta = read_y4m(v_path)
        im_dir = v_path[:-4]  # 分帧存放文件夹
        if not os.path.exists(im_dir):
            os.makedirs(im_dir)
        fid_len = len(str(len(frames) - 1))
        # save frame, todo(@izhx): also save meta?
        names = list()
        paths = list()
        for n, f in enumerate(frames):
            img_name = f"{vid}_{str(n).zfill(fid_len)}.npy"
            img_path = f"{im_dir}/{img_name}"
            np.save(img_path, f)
            names.append(img_name + '\n')
            paths.append(img_path[len(data_dir) + 1:] + '\n')
        with open(os.path.join(im_dir, "file_list.txt"), 'w') as f:
            f.writelines(names)
        with open(os.path.join(data_dir, "all_file_list.txt"), 'a+') as af:
            af.writelines(paths)
    else:
        t1 = time.time()
        print(f"\rSuccessful converted {len(videos)} videos in {t1 - t0:.4f} sec.", end="")
    return


if __name__ == '__main__':
    DIR = "../dataset/train"
    convert(DIR)
