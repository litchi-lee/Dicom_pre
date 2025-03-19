import glob
import os
import shutil

import cv2
import h5py
import imageio
import nibabel as nib
import numpy as np
import pydicom as dc
import scipy.ndimage
import SimpleITK as sitk
from natsort import natsorted
from PIL import Image, ImageOps
from tqdm import tqdm


# * 读取一个dicom文件，获取其基本信息
def dicom_pre(dicom_path):
    # 读取一个dicom文件，获取其基本信息
    ReferenceImage = dc.read_file(dicom_path)

    # 读取dicom文件的基本信息（三维信息）
    Dimension = (
        int(ReferenceImage.Rows),
        int(ReferenceImage.Columns),
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
    sliceID = ReferenceImage.data_element("InstanceNumber").value - 1

    print(f"Dimension: {Dimension}")
    print(f"Spacing: {Spacing}")
    print(f"Rescale_slope: {Rescale_slope}")
    print(f"Rescale_intercept: {Rescale_intercept}")
    print(f"Origin: {Origin}")
    print(f"sliceID: {sliceID}")


# * 只保留以keyword开头的文件名的png文件
def delete_unwanted_files(directory, keyword="2-"):
    # 获取目录下所有文件
    all_files = glob.glob(os.path.join(directory, '*'))

    # 遍历所有文件
    for file in all_files:
        # 如果文件名不是以“2-”开头，则删除
        if not os.path.basename(file).startswith(keyword):
            os.remove(file)

    all_files = natsorted(glob.glob(os.path.join(directory, '*')))

    for idx, file in enumerate(all_files):
        new_filename = f"{idx:03d}{os.path.splitext(file)[1]}"
        new_filepath = os.path.join(directory, new_filename)
        os.rename(file, new_filepath)


# * 用于保存CT扫描图像的不同切片视图（轴向、冠状和矢状）
def save_axis_image(ct_scan, root_path, prefix, slice_idx=None):
    if len(prefix) > 0:
        prefix = f"{prefix}_"

    # 保存不同切片视图的CT扫描图像
    for i, axis in zip([0, 1, 2], ['axial', 'coronal', 'sagittal']):
        if slice_idx is None:
            slice_idx = ct_scan.shape[i] // 2
        filename = os.path.join(
            root_path, f"{prefix}{axis}_output_gt_{slice_idx:0}.png"
        )
        if i == 0:
            slice_gt = ct_scan[slice_idx]
        elif i == 1:
            slice_gt = ct_scan[:, slice_idx]
        else:
            slice_gt = ct_scan[..., slice_idx]
        imageio.imwrite(filename, slice_gt)


# * 将.h5文件中的逐个切片转换为PNG图像
def convert_h5_to_png_slices(h5_dir, output_dir):
    # 获取所有 .h5 文件
    h5_files = glob.glob(os.path.join(h5_dir, '*.h5'))

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            ct_scan = np.array(f['ct'])

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        # 将每个切片保存为单独的PNG文件
        for i in tqdm(range(ct_scan.shape[0])):
            slice_data = ct_scan[i, :, :]

            # 将图像数据转换为 uint8 类型
            slice_data = (
                (slice_data - np.min(slice_data))
                / (np.max(slice_data) - np.min(slice_data))
                * 255
            )
            slice_data = slice_data.astype(np.uint8)

            # 设置文件名
            filename = os.path.join(output_dir, f"{i:03d}.png")

            # 保存为 .png 图像
            imageio.imwrite(filename, slice_data)

        print(f"Saved {output_dir}")


# * 将.dcm文件转换为PNG图像
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


# * 将.dcm文件裁剪后转换为PNG图像
def crop_dcm2png(dcm_path, output_dir="temp_test", crop_type=0):
    cropped_images = []

    df = dc.read_file(dcm_path)

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

    cropped_image = crop_image(image_pixel_array, crop_type=crop_type)

    # 将图像像素值整体加上1024，使其在0-2048之间
    cropped_image += 1024
    cropped_images.append(cropped_image)

    # 将裁剪后的图像列表转换为三维数组
    cropped_images = np.stack(cropped_images, axis=-1)
    cropped_images = np.transpose(cropped_images, (2, 0, 1))
    print(f"cropped_images.shape: {cropped_images.shape}")

    # 将图像数据转换为 uint8 类型
    image_array = cropped_images[0]
    image_array = (
        (image_array - np.min(image_array))
        / (np.max(image_array) - np.min(image_array))
        * 255
    )
    image_array = image_array.astype(np.uint8)

    # 获取文件名（不带扩展名）
    base_name = os.path.splitext(os.path.basename(dcm_path))[0] + "_" + str(crop_type)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存为 .png 图像
    filename = os.path.join(output_dir, f"{base_name}.png")
    imageio.imwrite(filename, image_array)
    print(f"Saved {filename}")


# * 裁剪CT图像
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


# * 从一批顺序文件中提取中间指定数量的文件，并将其按顺序存储在另一个路径
def extract_middle_files(input_dir, output_dir, num_files):
    """
    从一批顺序文件中提取中间指定数量的文件，并将其按顺序存储在另一个路径。

    参数:
    input_dir (str): 输入文件目录。
    output_dir (str): 输出文件目录。
    num_files (int): 要提取的文件数量。
    """
    # 获取所有文件，并按文件名排序
    all_files = sorted(glob.glob(os.path.join(input_dir, '*')))

    # 计算中间文件的起始和结束索引
    total_files = len(all_files)
    start_index = (total_files - num_files) // 2
    end_index = start_index + num_files

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 提取中间文件并复制到输出目录
    for idx, file in enumerate(tqdm(all_files[start_index:end_index])):
        new_filename = f"{idx:03d}{os.path.splitext(file)[1]}"
        new_filepath = os.path.join(output_dir, new_filename)
        shutil.copy(file, new_filepath)
    print(f"Saved {output_dir}")


# * 缩放图片
def resize_images(input_dir, output_dir, target_size):
    """
    批量缩放PNG格式的图片到目标分辨率。

    参数:
    input_dir (str): 输入图片目录。
    output_dir (str): 输出图片目录。
    target_size (tuple): 目标分辨率，格式为 (width, height)。
    """
    # 获取所有 .png 文件
    png_files = glob.glob(os.path.join(input_dir, '*.png'))

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for png_file in tqdm(png_files):
        # 打开图片
        img = Image.open(png_file)

        # 缩放图片
        img_resized = img.resize(target_size, Image.ANTIALIAS)

        # 获取文件名（不带扩展名）
        base_name = os.path.splitext(os.path.basename(png_file))[0]

        # 保存缩放后的图片
        output_path = os.path.join(output_dir, f"{base_name}_resized.png")
        img_resized.save(output_path)
        print(f"Saved {output_path}")


# * 反相灰度图片
def invert_images(input_dir, output_dir):
    """
    批量反相灰度图片。

    参数:
    input_dir (str): 输入图片目录。
    output_dir (str): 输出图片目录。
    """
    # 获取所有 .png 文件
    png_files = glob.glob(os.path.join(input_dir, '*.png'))

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for png_file in tqdm(png_files):
        # 打开图片
        img = Image.open(png_file)

        # 确保图像是灰度图像
        if img.mode != 'L':
            img = img.convert('L')

        # 反相图片
        img_inverted = ImageOps.invert(img)

        # 获取文件名（不带扩展名）
        base_name = os.path.splitext(os.path.basename(png_file))[0]

        # 保存反相后的图片
        output_path = os.path.join(output_dir, f"{base_name}_inverted.png")
        img_inverted.save(output_path)
        print(f"Saved {output_path}")


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


def resize_crop_image(image_root, new_size=(256, 256)):
    """
    调整图像的大小。

    参数:
    image (numpy.ndarray): 输入图像。
    new_size (tuple): 新的图像大小。

    返回:
    numpy.ndarray: 调整大小后的图像。
    """
    image_list = glob.glob(os.path.join(image_root, "*", "*", "*.png"))
    for image_path in image_list:
        image_save_path = image_path.replace(image_root, "tpdm_resized_xray")
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


# * 裁剪PNG格式的图片
def crop_png(input_dir, output_dir, crop_type=1):
    """
    批量裁剪PNG格式的图片。

    参数:
    input_dir (str): 输入图片目录。
    output_dir (str): 输出图片目录。
    crop_type (int): 裁剪类型。
    0: 左上角区域。
    1: 左中区域。
    2: 左下角区域。
    3: 右上角区域。
    4: 右中区域。
    5: 右下角区域。
    """
    # 获取所有 .png 文件
    png_files = glob.glob(os.path.join(input_dir, '*.png'))

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    for png_file in tqdm(png_files):
        # 打开图片
        img = Image.open(png_file)

        # 将PIL图像转换为NumPy数组
        img_array = np.array(img)

        # 裁剪图片
        img_cropped_array = crop_image(img_array, crop_type=crop_type)

        # 将裁剪后的NumPy数组转换回PIL图像
        img_cropped = Image.fromarray(img_cropped_array)

        # 保存裁剪后的图片
        output_path = png_file.replace(input_dir, output_dir)
        img_cropped.save(output_path)

    print(f"Saved {output_dir}")


def h5_to_nii(h5_path, nii_dir, tgt_res=128):
    """
    将 HDF5 数据保存为 NIfTI 文件。

    参数:
    h5_path (str): 输入 HDF5 文件路径。
    nii_path (str): 输出 NIfTI 文件路径。
    """
    # 读取 HDF5 数据
    with h5py.File(h5_path, "r") as f:
        image_data = f["ct"][:]
        affine = f["affine"][:] if "affine" in f else np.eye(4)

    # 调整图像大小
    target_shape = (tgt_res, tgt_res, tgt_res)
    zoom_factors = [t / s for t, s in zip(target_shape, image_data.shape)]
    resized_image_data = scipy.ndimage.zoom(image_data, zoom_factors, order=1)

    # 创建 NIfTI 对象
    nii_img = nib.Nifti1Image(resized_image_data, affine)

    # 保存为 .nii.gz
    nii_path = os.path.join(nii_dir, "ct.nii.gz")
    nib.save(nii_img, nii_path)


def create_video_from_images(image_dir, output_video_path, fps=10):
    """
    将一组图片合成为一个视频。

    参数:
    image_dir (str): 输入图片目录。
    output_video_path (str): 输出视频文件路径。
    fps (int): 视频帧率。
    """
    # 获取所有 .png 文件，并按自然顺序排序
    image_files = natsorted(glob.glob(os.path.join(image_dir, '*.png')))

    # 读取第一张图片以获取视频帧的宽度和高度
    first_image = cv2.imread(image_files[0])
    height, width, layers = first_image.shape

    # 创建视频写入对象
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 使用 mp4 编码
    video = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for image_file in image_files:
        img = cv2.imread(image_file)
        video.write(img)

    # 释放视频写入对象
    video.release()
    print(f"Video saved to {output_video_path}")


def convert_view(image_dir, output_dir, view='sagittal'):
    """
    将轴向视角的CT切片转换为矢状或冠状视角。

    参数:
    image_dir (str): 输入图片目录。
    output_dir (str): 输出图片目录。
    view (str): 目标视角，'sagittal' 或 'coronal'。
    """

    def load_images(image_dir):
        """
        加载一组图片并转换为三维NumPy数组。

        参数:
        image_dir (str): 输入图片目录。

        返回:
        numpy.ndarray: 三维图像数组。
        """
        image_files = natsorted(glob.glob(os.path.join(image_dir, '*.png')))
        images = [imageio.imread(image_file) for image_file in image_files]
        return np.stack(images, axis=0)

    def save_images(images, output_dir, prefix):
        """
        保存三维图像数组为一组PNG图片。

        参数:
        images (numpy.ndarray): 三维图像数组。
        output_dir (str): 输出图片目录。
        prefix (str): 文件名前缀。
        """
        os.makedirs(output_dir, exist_ok=True)
        for i in tqdm(range(images.shape[0])):
            image = images[i]
            filename = os.path.join(output_dir, f"{prefix}_{i:03d}.png")
            imageio.imwrite(filename, image)
            print(f"Saved {filename}")

    # 加载轴向视角的CT切片
    images = load_images(image_dir)

    # 转换视角
    if view == 'sagittal':
        converted_images = np.transpose(images, (2, 1, 0))
    elif view == 'coronal':
        converted_images = np.transpose(images, (1, 0, 2))
    else:
        raise ValueError("Invalid view: choose 'sagittal' or 'coronal'")

    # 保存转换后的切片
    save_images(converted_images, output_dir, view)


def sort_rename_files(source_dir_path):
    map_dict = {}

    dir_list = natsorted(glob.glob(os.path.join(source_dir_path, '*')))
    for idx, dir_ in enumerate(dir_list):
        dir_name = dir_.split(os.sep)[-1]
        map_dict[dir_name] = idx

        new_folder_name = f"ct{idx:0>4}"
        new_folder_path = os.path.join(source_dir_path, new_folder_name)

        # os.rename(dir_, new_folder_path)
    # print("dir rename done!")
    # print(map_dict)
    map_dict = dict(sorted(map_dict.items(), key=lambda x: x[1]))
    with open("map_dict.txt", "w") as f:
        f.write(str(map_dict))


if __name__ == "__main__":

    # ouput_root_dir = "ct_slice_png"
    # slice_list = glob.glob(os.path.join("4_ct_slice", "*", "*"))

    # for slice_path in tqdm(slice_list):
    #     ouput_dir = slice_path.replace("ct_slice", ouput_root_dir)
    #     os.makedirs(ouput_dir, exist_ok=True)
    #     convert_h5_to_png(slice_path, ouput_dir)

    # crop_dcm2png("0_data\\4400662\\2023-02-13\\3-72.dcm", "temp_test", crop_type=0)
    # crop_dcm2png("0_data\\4400662\\2023-02-13\\3-72.dcm", "temp_test", crop_type=1)
    # crop_dcm2png("0_data\\4400662\\2023-02-13\\3-72.dcm", "temp_test", crop_type=2)
    # crop_dcm2png("0_data\\4400662\\2023-02-13\\3-72.dcm", "temp_test", crop_type=3)
    # crop_dcm2png("0_data\\4400662\\2023-02-13\\3-72.dcm", "temp_test", crop_type=4)
    # crop_dcm2png("0_data\\4400662\\2023-02-13\\3-72.dcm", "temp_test", crop_type=5)

    # source_h5_dir = "3_ctpro_hdf5"
    # source_h5_path_list = glob.glob(os.path.join(source_h5_dir, "*"))
    # for source_h5_path in source_h5_path_list:
    #     output_dir = source_h5_path.replace(source_h5_dir, "3_ctpro_png")
    #     os.makedirs(output_dir, exist_ok=True)
    #     convert_h5_to_png_slices(source_h5_path, output_dir)

    # source_png_dir = "3_ctpro_png"
    # source_png_path_list = glob.glob(os.path.join(source_png_dir, "*"))
    # for source_png_path in source_png_path_list:
    #     output_dir = source_png_path.replace(source_png_dir, "3_ctpro_png_extract")
    #     os.makedirs(output_dir, exist_ok=True)
    #     extract_middle_files(source_png_path, output_dir, 256)

    # * tpdm处理
    # source_png_dir = "tpdm_data"
    # source_png_path_list = glob.glob(os.path.join(source_png_dir, "*", "*"))
    # keyword_list = ["202-", "302-"]
    # for source_png_path, keyword in zip(source_png_path_list, keyword_list):
    #     delete_unwanted_files(source_png_path, keyword=keyword)

    # source_png_dir = "tpdm_data"
    # source_png_path_list = glob.glob(os.path.join(source_png_dir, "*", "*"))
    # for source_png_path in source_png_path_list:
    #     output_dir = source_png_path.replace(source_png_dir, "tpdm_png_extract")
    #     os.makedirs(output_dir, exist_ok=True)
    #     extract_middle_files(source_png_path, output_dir, 256)

    # source_png_dir = "tpdm_png_extract"
    # source_png_path_list = glob.glob(os.path.join(source_png_dir, "*", "*"))
    # crop_type_list = [4, 4]
    # for source_png_path, crop_type in zip(source_png_path_list, crop_type_list):
    #     output_dir = source_png_path.replace(source_png_dir, "tpdm_png_cropped")
    #     os.makedirs(output_dir, exist_ok=True)
    #     crop_png(source_png_path, output_dir, crop_type=crop_type)

    # source_xray_dir = "tpdm_xray"
    # resize_crop_image(source_xray_dir, new_size=(256, 256))

    # source_xray_dir = "tpdm_resized_xray"
    # source_xray_path_list = glob.glob(os.path.join(source_xray_dir, "*", "*"))
    # for source_xray_path in source_xray_path_list:
    #     output_dir = source_xray_path.replace(source_xray_dir, "tpdm_xray_inverted")
    #     os.makedirs(output_dir, exist_ok=True)
    #     invert_images(source_xray_path, output_dir)

    # source_h5_dir = "D:\\datasets\\lidc_idri\\LIDC-HDF5-256_ct320_ct128"
    # source_h5_path_list = glob.glob(os.path.join(source_h5_dir, "*"))
    # for source_h5_path in tqdm(source_h5_path_list):
    #     source_h5 = glob.glob(os.path.join(source_h5_path, "*.h5"))[0]
    #     output_dir = source_h5_path.replace(
    #         "LIDC-HDF5-256_ct320_ct128", "LIDC-IDRI-XctDiff"
    #     )
    #     os.makedirs(output_dir, exist_ok=True)
    #     h5_to_nii(source_h5, output_dir)

    # image_dir = "tpdm_png_cropped\\4673278\\2023-09-10"
    # output_video_path = "tpdm_video2.mp4"
    # create_video_from_images(image_dir, output_video_path, fps=10)

    # image_dir1 = "tpdm_png_cropped\\4673278\\2023-09-10"
    # image_dir2 = "tpdm_png_cropped\\4841102\\2018-08-20"
    # output_dir_sagittal1 = "ct_png_sagittal\\4673278"
    # output_dir_sagittal2 = "ct_png_sagittal\\4841102"
    # convert_view(image_dir1, output_dir_sagittal1, view='sagittal')
    # convert_view(image_dir2, output_dir_sagittal2, view='sagittal')

    sort_rename_files("D:/datasets/lidc_idri/LIDC-IDRI-XctDiff")
