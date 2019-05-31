import gc
import os
import glob
import time
import numpy as np
from data.youku import YoukuDataset


def read_y4m(file_path):
    """
    读取y4m视频文件为4维ndarray

    :param file_path: the path of a y4m file (420planar).

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

    for frame in yuv_frames:
        y = np.array((frame[:end_y]), dtype=np.uint8).reshape(height, width)
        u = np.array(frame[start_u:end_u], dtype=np.uint8).reshape(uvh, uvw)
        v = np.array(frame[start_v:end_v], dtype=np.uint8).reshape(uvh, uvw)
        u, v = extend_uv(u), extend_uv(v)
        frames.append(np.array([y, u, v]).transpose((1, 2, 0)))

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


if __name__ == '__main__':
    yk = YoukuDataset("../dataset/train", 4, 5, True, True, 31, "new_info")
    DIR = "../dataset/train"
    convert(DIR)

    # header: 'signature width height fps interlacing pixelAspectRadio colorSpace comment'
