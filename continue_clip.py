import cv2 as cv
import numpy as np
from skimage.measure import compare_ssim


def matches(current, next):
    queryImage = current
    trainingImage = next  # 读取要匹配的灰度照片
    sift = cv.xfeatures2d.SIFT_create()  # 创建sift检测器
    kp1, des1 = sift.detectAndCompute(queryImage, None)
    kp2, des2 = sift.detectAndCompute(trainingImage, None)
    FLANN_INDEX_KDTREE = 0
    indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    searchParams = dict(checks=50)
    flann = cv.FlannBasedMatcher(indexParams, searchParams)
    # cv.imshow("queryImage", queryImage)
    # cv.imshow("trainingImage", trainingImage)
    if len(kp1) != 1 and len(kp2) != 1 and kp1 and kp2:  # 特征点个数不能为1
        matches = flann.knnMatch(des1, des2, k=2)
    else:
        return 0
    matchesMask = [[0, 0] for i in range(len(matches))]
    sum = 0  # 匹配成功的点数
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:  # 判断是否匹配
            if kp1[i].pt[1] < 210:
                matchesMask[i] = [1, 0]
                sum += 1
    return sum


def is_continue_(current, next):
    similarity = compare_ssim(current, next)
    if similarity > 0.95:
        return True
    else:
        return False


def log_(file_, **massage):
    condition_ = massage
    for k in massage.keys():
        file_.write(k + ":{}\n".format(str(massage[k])))
    return condition_
    # if warning_:
    #     file_.write("##############warning##############\n")


def is_continue_withlog(current, next, log_file):
    sum_current_matches = matches(current, current)
    sum_next_matches = matches(current, next)
    H1 = cv.calcHist([current], [0], None, [256], [0, 256])
    H1 = cv.normalize(H1, H1, 0, 1, cv.NORM_MINMAX, -1)  # 对图片进行归一化处理

    H2 = cv.calcHist([next], [0], None, [256], [0, 256])  # 计算图img2的直方图
    H2 = cv.normalize(H2, H2, 0, 1, cv.NORM_MINMAX, -1)

    similarity = cv.compareHist(H1, H2, 0)  # 利用compareHist（）进行比较相似度
    similarity_ = compare_ssim(current, next)
    if sum_next_matches < int(sum_current_matches * 0.1):  # and sum_current_matches >= 90:#如果特征充足且未匹配上，则不连续
        # H1 = cv.calcHist([current], [0], None, [256], [0, 256])
        # H1 = cv.normalize(H1, H1, 0, 1, cv.NORM_MINMAX, -1)  # 对图片进行归一化处理
        #
        # H2 = cv.calcHist([n ext], [0], None, [256], [0, 256])  # 计算图img2的直方图
        # H2 = cv.normalize(H2, H2, 0, 1, cv.NORM_MINMAX, -1)
        #
        # similarity = cv.compareHist(H1, H2, 0)  # 利用compareHist（）进行比较相似度
        # similarity_ = compare_ssim(current, next)
        if similarity + similarity_ > 1.7:
            print(sum_current_matches, sum_next_matches)
            print(similarity)
            print(similarity_)
            condition = log_(log_file, cut_class="Warning",
                             sum_current_matches=sum_current_matches,
                             sum_next_matches=sum_next_matches,
                             similarity=similarity,
                             similarity_=similarity_)
            print("##############warning##############")
            return True, condition
        print(sum_current_matches, sum_next_matches)
        print(similarity)
        print(similarity_)
        condition = log_(log_file, cut_class="Normal",
                         sum_current_matches=sum_current_matches,
                         sum_next_matches=sum_next_matches,
                         similarity=similarity,
                         similarity_=similarity_)
        return False, condition
    elif sum_next_matches < int(matches(next, next) * 0.1):
        print(sum_current_matches, sum_next_matches, matches(next, next))
        print(similarity)
        print(similarity_)
        condition = log_(log_file, cut_class="Reversed_Normal",
                         sum_current_matches=sum_current_matches,
                         sum_next_matches=sum_next_matches,
                         similarity=similarity,
                         similarity_=similarity_)
        return False, condition
    # elif abs(-sum_current_matches)/sum_current_matches> 0.8:
    #     print(sum_current_matches, sum_next_matches, matches(next, next), 2)
    #     return False
    # elif sum_next_matches < sum_current_matches*0.05:#如果特征未配上或者特征不足，
    #     print(sum_current_matches, sum_next_matches, 3)
    #     return False
    else:
        # print(similarity)
        condition = log_(log_file, cut_class="Continue",
                         sum_current_matches=sum_current_matches,
                         sum_next_matches=sum_next_matches,
                         similarity=similarity,
                         similarity_=similarity_)
        return True, condition


# 以下为接口函数
def is_continue(current, next):  # 判断两帧是否连续，输入两帧
    sum_current_matches = matches(current, current)
    sum_next_matches = matches(current, next)
    if sum_next_matches < int(sum_current_matches * 0.1):  # and sum_current_matches >= 90:#如果特征充足且未匹配上，则不连续
        H1 = cv.calcHist([current], [0], None, [256], [0, 256])
        H1 = cv.normalize(H1, H1, 0, 1, cv.NORM_MINMAX, -1)  # 对图片进行归一化处理

        H2 = cv.calcHist([next], [0], None, [256], [0, 256])  # 计算图img2的直方图
        H2 = cv.normalize(H2, H2, 0, 1, cv.NORM_MINMAX, -1)

        similarity = cv.compareHist(H1, H2, 0)  # 利用compareHist（）进行比较相似度
        similarity_ = compare_ssim(current, next)
        if similarity + similarity_ > 1.7:
            return True
        return False
    elif sum_next_matches < int(matches(next, next) * 0.1):
        return False
    else:
        return True


def continue_frames(yuv_imgs):  # 输入重建帧及其参考帧，根据是否连续进行补全，例如七帧或五帧
    grey_imgs = yuv_imgs[..., 0]
    imgs_len = len(grey_imgs)
    for i in range(imgs_len // 2):
        if is_continue(grey_imgs[imgs_len // 2 + i], grey_imgs[imgs_len // 2 + i + 1]):
            pass
        else:
            yuv_imgs[imgs_len // 2 + i + 1] = yuv_imgs[imgs_len // 2 + i]
        if is_continue(grey_imgs[imgs_len // 2 - i], grey_imgs[imgs_len // 2 - i - 1]):
            pass
        else:
            yuv_imgs[imgs_len // 2 - i - 1] = yuv_imgs[imgs_len // 2 - i]
    return yuv_imgs  # 返回补全后结果


def cut_clips(yuv_video):  # 将视频切成片段
    gery_imgs = yuv_video[..., 0]
    gery_imgs.dtype = np.uint8
    continue_frames = []
    continue_ = []
    cuts = []
    for j, img in enumerate(gery_imgs):
        is_continue_, logs = is_continue(gery_imgs[j - 1], img)
        if len(continue_) == 0:
            cuts.append(0)
            continue_.append(yuv_video[j])
        elif is_continue_:
            continue_.append(yuv_video[j])
            if j == len(gery_imgs) - 1:
                continue_frames.append(continue_)
        else:
            if j == len(gery_imgs) - 1:
                continue_frames.append(continue_)
                continue_ = [yuv_video[j]]
                cuts.append(j)
                continue_frames.append(continue_)
            else:
                continue_frames.append(continue_)
                continue_ = [yuv_video[j]]
                cuts.append(j)
    return continue_frames, cuts  # 第一个返回值是片段列表，第二个是每个片段第一帧的帧数
