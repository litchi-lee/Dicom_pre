import glob
import os

import h5py
import imageio
import numpy as np
import pydicom as dc
import SimpleITK as sitk
from tqdm import tqdm


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


# 只保留以keyword开头的文件名的文件
def delete_unwanted_files(directory, keyword="2-"):
    # 获取目录下所有文件
    all_files = glob.glob(os.path.join(directory, '*'))

    # 遍历所有文件
    for file in all_files:
        # 如果文件名不是以“2-”开头，则删除
        if not os.path.basename(file).startswith(keyword):
            os.remove(file)


# 用于保存CT扫描图像的不同切片视图（轴向、冠状和矢状）
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


def convert_h5_to_png(h5_dir, output_dir):
    # 获取所有 .h5 文件
    h5_files = glob.glob(os.path.join(h5_dir, '*.h5'))

    for h5_file in h5_files:
        with h5py.File(h5_file, 'r') as f:
            ct_scan = np.array(f['ct'])

        # 获取文件名（不带扩展名）
        base_name = os.path.splitext(os.path.basename(h5_file))[0]

        # 将图像数据转换为 uint8 类型
        ct_scan = (
            (ct_scan - np.min(ct_scan)) / (np.max(ct_scan) - np.min(ct_scan)) * 255
        )
        ct_scan = ct_scan.astype(np.uint8)

        filename = os.path.join(output_dir, f"{base_name}.png")
        imageio.imwrite(filename, ct_scan)


if __name__ == "__main__":
    ouput_root_dir = "ct_slice_png"
    slice_list = glob.glob(os.path.join("ct_slice", "*", "*"))

    for slice_path in tqdm(slice_list):
        ouput_dir = slice_path.replace("ct_slice", ouput_root_dir)
        os.makedirs(ouput_dir, exist_ok=True)
        convert_h5_to_png(slice_path, ouput_dir)
