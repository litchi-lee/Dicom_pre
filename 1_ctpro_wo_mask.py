import glob
import os
import time

import numpy as np
import scipy.ndimage as ndimage

from ctxray_utils import load_scan_mhda, save_scan_mhda


# 将图像重采样到新的像素间距，以保持物理尺度稳定，同时改变像素数量
# image: 输入图像
# spacing: 原始像素间距
# new_spacing: 新的像素间距
def resample(image, spacing, new_spacing=[1, 1, 1]):
    # .mhd image order : z, y, x
    if not isinstance(spacing, np.ndarray):
        spacing = np.array(spacing)
    if not isinstance(new_spacing, np.ndarray):
        new_spacing = np.array(new_spacing)
    spacing = spacing[::-1]
    new_spacing = new_spacing[::-1]
    spacing = np.array(list(spacing))

    # 计算重采样因子，并根据此计算新的图像形状
    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor

    # 使用scipy的zoom函数进行重采样
    image = ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')

    return image, new_spacing


# 该函数将图像裁剪到标准形状（scale x scale x scale），以保持像素尺度一致性
# scan: 输入图像
# scale: 标准形状的大小
def crop_to_standard(scan, scale):
    z, y, x = scan.shape

    # 处理深度（z轴）
    # 如果深度大于或等于目标形状，则从输入图像的末尾截取目标形状的深度部分
    if z >= scale:
        ret_scan = scan[z - scale : z, :, :]
    # 如果深度小于目标形状，则在输入图像的两端填充0，使其深度等于目标形状
    else:
        temp1 = np.zeros(((scale - z) // 2, y, x))
        temp2 = np.zeros(((scale - z) - (scale - z) // 2, y, x))
        ret_scan = np.concatenate((temp1, scan, temp2), axis=0)
    z, y, x = ret_scan.shape

    # 处理深度（y轴）
    # 如果高度大于或等于目标形状，则从输入图像的中间截取目标形状的高度部分
    if y >= scale:
        ret_scan = ret_scan[:, (y - scale) // 2 : (y + scale) // 2, :]
    # 如果高度小于目标形状，则在输入图像的两端填充0，使其高度等于目标形状
    else:
        temp1 = np.zeros((z, (scale - y) // 2, x))
        temp2 = np.zeros((z, (scale - y) - (scale - y) // 2, x))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=1)
    z, y, x = ret_scan.shape

    # 处理深度（x轴）
    # 如果宽度大于或等于目标形状，则从输入图像的中间截取目标形状的宽度部分
    if x >= scale:
        ret_scan = ret_scan[:, :, (x - scale) // 2 : (x + scale) // 2]
    # 如果宽度小于目标形状，则在输入图像的两端填充0，使其宽度等于目标形状
    else:
        temp1 = np.zeros((z, y, (scale - x) // 2))
        temp2 = np.zeros((z, y, (scale - x) - (scale - x) // 2))
        ret_scan = np.concatenate((temp1, ret_scan, temp2), axis=2)

    return ret_scan


# 对一组CT扫描进行处理，将其重采样到新的像素间距，然后裁剪到标准形状
# root_dir: 数据集根目录
# input_subdir: 输入CT扫描的子目录
# output_subdir: 输出CT扫描的子目录
# new_spacing: 新的像素间距 默认为[1, 1, 1]
# scale: 标准形状的大小 默认为320
def process_ct_scans(
    root_dir, input_subdir, output_subdir, new_spacing=[1, 1, 1], scale=320
):
    # 获取所有.mhd文件的路径
    root_path = os.path.join(root_dir, input_subdir)
    save_root_path = os.path.join(root_dir, output_subdir)
    files_list = glob.glob(os.path.join(root_path, '*.mhd'))
    files_list = sorted(files_list)

    start = time.time()
    # 对每一个.mhd文件进行处理
    for fileIndex, filePath in enumerate(files_list):
        t0 = time.time()
        # 获取文件名
        _, file = os.path.split(filePath)
        fileName = os.path.splitext(file)[0]
        print('Begin {}/{}: {}'.format(fileIndex + 1, len(files_list), fileName))

        # 创建保存目录
        saveDir = os.path.join(save_root_path, fileName)
        os.makedirs(saveDir, exist_ok=True)
        savePath = os.path.join(saveDir, '{}.mha'.format('ct_file'))

        # 加载原始CT扫描
        ct_itk, ct_scan, ori_origin, ori_size, ori_spacing = load_scan_mhda(filePath)
        print("Old : ", ori_size)

        # 重采样到新的像素间距
        bedstrip_scan = ct_scan
        new_scan, new_spacing = resample(bedstrip_scan, ori_spacing, new_spacing)
        print("Std : ", new_scan.shape[::-1])

        # 裁剪到标准形状
        std_scan = crop_to_standard(new_scan, scale)

        # 保存处理后的CT扫描
        save_scan_mhda(std_scan, (0, 0, 0), new_spacing, savePath)

        # 加载保存的CT扫描，检查是否保存正确
        _, _, _, new_size, new_spacing = load_scan_mhda(savePath)
        print("New : ", new_size)

        t1 = time.time()
        print('End! Case time: {}'.format(t1 - t0))

    end = time.time()
    print('Finally! Total time: {}'.format(end - start))


if __name__ == '__main__':
    # 示例调用（需要修改root_dir、input_subdir、output_subdir）
    root_dir = os.path.abspath("D:/datasets")
    input_subdir = "lidc_idri/LIDC-IDRI-MDH"
    output_subdir = "lidc_idri/LIDC-IDRI-MDH_ctpro"
    new_spacing = [1, 1, 1]
    scale = 320
    process_ct_scans(root_dir, input_subdir, output_subdir, new_spacing, scale)
