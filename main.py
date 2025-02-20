import glob
import os
import sys
import time

import cv2
import h5py
import imageio
import numpy as np
import pydicom as dc
import scipy.ndimage as ndimage
import SimpleITK as sitk
from PIL import Image
from scipy.ndimage import zoom
from tqdm import tqdm

from ctxray_utils import load_scan_mhda, save_scan_mhda


def dicom_to_mhd(
    folders_path, save_path, text_file_name="dicom2mhd.txt", crop_type_list=[0]
):

    text_path = os.path.join(os.path.dirname(save_path), text_file_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(text_path, 'w') as file:
        file.write(
            "filename\tRescaleSlope\tRescaleIntercept\tDimension\tSpacing\tOrigin\n"
        )

    dir_list = glob.glob(os.path.join(folders_path, '*', '*'))
    print(f"提取的dicom示例(0号): {dir_list[0]}")
    for dir_, crop_type in tqdm(zip(dir_list, crop_type_list)):
        files = glob.glob(os.path.join(dir_, '*.dcm'))
        if len(files) > 10:
            # 目录和文件名处理
            dir_names = dir_[len(folders_path) :].split(os.sep)
            # print(f"dir_names: {dir_names}")
            save_fileName = dir_names[1]
            # print(f"save_fileName: {save_fileName}")

            # Dicom文件列表处理
            DICOM_LIST = sorted(files)
            DICOM_LIST = [x.strip() for x in DICOM_LIST]
            DICOM_LIST = [x[0 : x.find(".dcm") + 4] for x in DICOM_LIST]

            # 读取第一个dicom文件，获取基本信息并将其作为参考
            ReferenceImage = dc.read_file(DICOM_LIST[0])
            with open(f"{save_path}\\{save_fileName}.txt", 'w') as file:
                file.write(f"{ReferenceImage}")

            # 读取dicom文件的基本信息（三维信息）
            Dimension = (
                int(ReferenceImage.Rows),
                int(ReferenceImage.Columns),
                len(DICOM_LIST),
            )

            # 读取dicom文件的像素间距、重采样斜率、重采样截距、原点
            Spacing = (
                float(ReferenceImage.PixelSpacing[0]),
                float(ReferenceImage.PixelSpacing[1]),
                float(ReferenceImage.SliceThickness),
            )
            Rescale_slope = ReferenceImage.data_element("RescaleSlope").value
            Rescale_intercept = ReferenceImage.data_element("RescaleIntercept").value
            Origin = ReferenceImage.ImagePositionPatient

            sliceIDs = []
            for filename in DICOM_LIST:
                image = dc.read_file(filename)
                sliceID = image.data_element("InstanceNumber").value - 1
                sliceIDs.append(sliceID)

            sliceIDs.sort()
            cropped_images = []
            for filename in DICOM_LIST:
                df = dc.read_file(filename)
                # 获取图像维度、间距、原点、重采样斜率、重采样截距
                dimension = [int(df.Rows), int(df.Columns), len(df)]
                spacing = [
                    float(df.PixelSpacing[0]),
                    float(df.PixelSpacing[1]),
                    float(df.SliceThickness),
                ]
                origin = df.ImagePositionPatient
                rescale_slope = df.data_element("RescaleSlope").value
                rescale_intercept = df.data_element("RescaleIntercept").value

                # 对图像像素重缩放得到实际的图像像素值
                image_pixel_array = df.pixel_array * rescale_slope + rescale_intercept

                # 获取当前图像的切片ID
                sliceID = df.data_element("InstanceNumber").value - 1
                sliceID = sliceIDs.index(sliceID)

                cropped_image = crop_image(image_pixel_array, crop_type=crop_type)

                # 将图像像素值整体加上1024，使其在0-2048之间
                cropped_image += 1024
                cropped_images.append(cropped_image)

            # 将裁剪后的图像列表转换为三维数组
            cropped_images = np.stack(cropped_images, axis=-1)
            with open(text_path, 'a') as file:
                file.write(
                    f"{save_fileName}\t{rescale_slope}\t{rescale_intercept}\t{dimension}\t{spacing}\t{origin}\n"
                )

            cropped_images = np.transpose(cropped_images, (2, 0, 1))
            sitk_img = sitk.GetImageFromArray(cropped_images, isVector=False)
            sitk_img.SetSpacing(Spacing)
            sitk_img.SetOrigin(Origin)
            sitk.WriteImage(sitk_img, os.path.join(save_path, save_fileName + ".mhd"))
    print("dicom2mhd done!")


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
def process_ct_scans(root_path, save_root_path, new_spacing=[1, 1, 1], scale=320):
    # 获取所有.mhd文件的路径
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


def crop_image(image, crop_type=0):
    """
    裁剪图像的左上角区域（左上的四分之一区域）。

    参数:
    image (numpy.ndarray): 输入图像。
    crop_type (int): 裁剪类型。默认为0。
    0: 左上角区域。
    1: 左中区域。
    2: 左下角区域。
    3: 右上角区域。
    4: 右中区域。
    5: 右下角区域。

    返回:
    numpy.ndarray: 裁剪后的图像。
    """
    height, width = image.shape
    crop_fraction = 0.5
    cropped_height = int(height * crop_fraction)
    cropped_width = int(width * crop_fraction)

    if crop_type == 0:
        return image[:cropped_height, :cropped_width]
    elif crop_type == 1:
        return image[height // 4 : height // 4 + cropped_height, :cropped_width]
    elif crop_type == 2:
        return image[cropped_height:width, :cropped_width]
    elif crop_type == 3:
        return image[:cropped_height, cropped_width:width]
    elif crop_type == 4:
        return image[height // 4 : height // 4 + cropped_height, cropped_width:width]
    elif crop_type == 5:
        return image[cropped_height:height, cropped_width:width]


def convert_dcm_to_png(dcm_dir, output_dir):
    # 获取所有 .dcm 文件
    dcm_files = glob.glob(os.path.join(dcm_dir, '*.dcm'))

    for dcm_file in dcm_files:
        # 读取 DICOM 文件
        ds = dc.dcmread(dcm_file)

        # 获取像素数组
        image_array = ds.pixel_array

        # 将图像数据转换为 uint8 类型
        image_array = (
            (image_array - np.min(image_array))
            / (np.max(image_array) - np.min(image_array))
            * 255
        )
        image_array = image_array.astype(np.uint8)

        # 获取文件名（不带扩展名）
        base_name = os.path.splitext(os.path.basename(dcm_file))[0]

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 保存为 .png 图像
        filename = os.path.join(output_dir, f"{base_name}.png")
        imageio.imwrite(filename, image_array)
        print(f"Saved {filename}")


def crop_center_square(image):
    """
    从图像的中间部分裁剪出一个正方形。

    参数:
    image (numpy.ndarray): 输入图像。

    返回:
    numpy.ndarray: 裁剪后的正方形图像。
    """
    height, width = image.shape
    min_dim = min(height, width)
    start_x = (width - min_dim) // 2
    start_y = (height - min_dim) // 2
    return image[start_y : start_y + min_dim, start_x : start_x + min_dim]


def resize_image(image_root, new_size=(320, 320)):
    """
    调整图像的大小。

    参数:
    image (numpy.ndarray): 输入图像。
    new_size (tuple): 新的图像大小。

    返回:
    numpy.ndarray: 调整大小后的图像。
    """
    image_list = glob.glob(os.path.join(image_root, "*", "*.png"))
    for image_path in image_list:
        image_save_path = image_path.replace("xrays", "ctpro")
        save_dir = os.path.dirname(image_save_path)

        # 如果保存路径中的文件夹不存在，则创建它
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        cropped_image = crop_center_square(image)
        resized_image = cv2.resize(
            cropped_image, new_size, interpolation=cv2.INTER_AREA
        )
        cv2.imwrite(image_save_path, resized_image)
        print(f"Resized image saved to {image_save_path}")


# 用于创建HDF5数据集，并保存CT扫描图像的不同切片视图（轴向、冠状和矢状）
def make_h5_dataset_LIDC(mda_root_path, save_ct_path, save_xray_path, xray_size=128):

    if not os.path.exists(save_ct_path):
        os.makedirs(save_ct_path)

    if not os.path.exists(f"{save_xray_path}"):
        os.makedirs(f"{save_xray_path}")

    # 获取源路径中的所有文件夹
    dir_list = glob.glob(os.path.join(mda_root_path, "*"))
    print(f"Number of data in source path : {len(dir_list)}")

    # dir_ = "ctpro/4063265"
    for dir_ in tqdm(dir_list):
        key = dir_.split(os.sep)[-1]
        save_subfolder_path = os.path.join(save_ct_path, key)
        if not os.path.exists(save_subfolder_path):
            os.makedirs(save_subfolder_path)  ## ct_xray_data.h5
        ct_path = glob.glob(os.path.join(dir_, '*.mha'))[0]

        _, ct_scan, _, _, _ = load_scan_mhda(ct_path)
        save_name = os.path.join(save_subfolder_path, "ct_xray_data.h5")
        f = h5py.File(save_name, 'w')
        f['ct'] = ct_scan
        f.close()

        xray_direction1 = cv2.imread(
            glob.glob(os.path.join(dir_, '*.png'))[0], cv2.IMREAD_GRAYSCALE
        )
        xray_direction2 = cv2.imread(
            glob.glob(os.path.join(dir_, '*.png'))[1], cv2.IMREAD_GRAYSCALE
        )

        if xray_direction1.shape[1] != xray_size:
            xray_direction1 = Image.fromarray(xray_direction1)
            xray_direction1 = xray_direction1.resize(
                (xray_size, xray_size), Image.LANCZOS
            )
            xray_direction1 = np.array(xray_direction1)
        if xray_direction2.shape[1] != xray_size:
            xray_direction2 = Image.fromarray(xray_direction2)
            xray_direction2 = xray_direction2.resize(
                (xray_size, xray_size), Image.LANCZOS
            )
            xray_direction2 = np.array(xray_direction2)

        cv2.imwrite(
            os.path.join(save_xray_path, f"{key}_xray1.png"),
            xray_direction1,
        )
        cv2.imwrite(
            os.path.join(save_xray_path, f"{key}_xray2.png"),
            xray_direction2,
        )


sys.path.append(os.getcwd())


# 检查是否是数组
def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


# 重置图像大小
class Resize_image(object):
    '''
    Returns:
      img: 3d array, (z,y,x) or (D, H, W)
    '''

    def __init__(self, size=(3, 256, 256)):
        if not _isArrayLike(size):
            raise ValueError('each dimension of size must be defined')
        self.size = np.array(size, dtype=np.float32)

    def __call__(self, img):
        z, x, y = img.shape
        ori_shape = np.array((z, x, y), dtype=np.float32)
        resize_factor = self.size / ori_shape
        img_copy = zoom(img, resize_factor, order=1)

        return img_copy


def ct_preprocessing(root_path, trg_ct_res, Save_path):
    '''
    root_path: str, path to the data
    trg_ct_res: int, target resolution
    Save_path: str, path to save the processed data
    '''
    base_ct_res = [320, 320, 320]
    file_name = "ct_xray_data.h5"
    dir_list = glob.glob(os.path.join(root_path, '*'))

    # 遍历.h5文件目录
    for dir in tqdm(dir_list):
        file_path = os.path.join(dir, file_name)  # ctpro_hdf5/4063265/ct_xray_data.h5
        save_file_name = file_name

        sub_folder = os.path.basename(dir)  # 4063265
        ct_save_path = os.path.join(Save_path, sub_folder, "ct")  # ctslice/4063265/ct

        # 读取CT扫描
        hdf5 = h5py.File(file_path, 'r')
        ori_ct = np.asarray(hdf5['ct'])
        hdf5.close()
        os.makedirs(ct_save_path, exist_ok=True)

        for slice_idx in range(trg_ct_res):
            if ori_ct.shape[0] != trg_ct_res:
                ct_res = base_ct_res.copy()
                ct_res[0] = trg_ct_res
                ct = Resize_image(size=ct_res)(ori_ct)
            else:
                ct = ori_ct.copy()
            slice_gt = ct[slice_idx]
            filename = os.path.join(ct_save_path, f"axial_{slice_idx:0>3}.h5")
            f = h5py.File(filename, 'w')
            f['ct'] = slice_gt
            f.close()

        for slice_idx in range(trg_ct_res):
            if ori_ct.shape[1] != trg_ct_res:
                ct_res = base_ct_res.copy()
                ct_res[1] = trg_ct_res
                ct = Resize_image(size=ct_res)(ori_ct)
            else:
                ct = ori_ct.copy()
            slice_gt = ct[:, slice_idx]
            filename = os.path.join(ct_save_path, f"coronal_{slice_idx:0>3}.h5")
            f = h5py.File(filename, 'w')
            f['ct'] = slice_gt
            f.close()

        for slice_idx in range(trg_ct_res):
            if ori_ct.shape[2] != trg_ct_res:
                ct_res = base_ct_res.copy()
                ct_res[2] = trg_ct_res
                ct = Resize_image(size=ct_res)(ori_ct)
            else:
                ct = ori_ct.copy()
            slice_gt = ct[..., slice_idx]
            filename = os.path.join(ct_save_path, f"sagittal_{slice_idx:0>3}.h5")
            f = h5py.File(filename, 'w')
            f['ct'] = slice_gt
            f.close()


if __name__ == "__main__":
    # 4063265 R 取镜像左边
    # 4400662 L 取镜像右边

    # 1. 将DICOM文件转换为MHD文件
    # dicom_to_mhd(folders_path="data", save_path="mhd", crop_type_list=[0, 4])

    # 2. 将CT扫描重采样到新的像素间距，然后裁剪到标准形状
    # process_ct_scans(root_path="mhd", save_root_path="ctpro")

    # 3. 将X光图像裁剪到标准形状
    # resize_image(image_root="xrays", new_size=(320, 320))

    # 4. 创建HDF5数据集
    # make_h5_dataset_LIDC(
    #     mda_root_path="ctpro",
    #     save_ct_path="ctpro_hdf5",
    #     save_xray_path="xrays_hdf5",
    #     xray_size=128,
    # )

    # 5. resize CT Xray
    ct_preprocessing("ctpro_hdf5", 128, "ct_slice")
