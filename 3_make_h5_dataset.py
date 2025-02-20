import glob
import os
import pdb
import shutil

import cv2
import h5py
import imageio
import numpy as np
from PIL import Image
from tqdm import tqdm

from ctxray_utils import load_scan_mhda


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


# 用于创建HDF5数据集，并保存CT扫描图像的不同切片视图（轴向、冠状和矢状）
def make_h5_dataset_LIDC(root_path, mda_path, save_path, save_Xray_path, xray_size=128):

    # 设置路径
    root_dir = os.path.abspath(root_path)
    source_Path = os.path.join(root_dir, mda_path)
    save_ct_path = os.path.join(root_dir, save_path)
    save_xray_path = os.path.join(root_dir, save_Xray_path)

    if not os.path.exists(save_ct_path):
        os.makedirs(save_ct_path)

    if not os.path.exists(f"{save_xray_path}"):
        os.makedirs(f"{save_xray_path}")

    # 获取源路径中的所有文件夹
    dir_list = glob.glob(os.path.join(source_Path, "*"))
    print(f"Number of data in source path : {len(dir_list)}")

    # 读取txt文件中的数据，分别得到训练、测试和验证数据
    with open(
        os.path.join(
            os.path.abspath("D:/datasets/lidc_idri/dataset_list"), "train.txt"
        ),
        'r',
    ) as file:
        train_list = file.readlines()
    with open(
        os.path.join(os.path.abspath("D:/datasets/lidc_idri/dataset_list"), "test.txt"),
        'r',
    ) as file:
        test_list = file.readlines()
    with open(
        os.path.join(os.path.abspath("D:/datasets/lidc_idri/dataset_list"), "val.txt"),
        'r',
    ) as file:
        val_list = file.readlines()

    total_list = train_list + test_list + val_list
    total_list.sort()
    print(f"Number of data in txt files : {len(total_list)}")

    total_list_dict = {}
    for t in total_list:
        key = t.split("_")[0]
        value = t.split("\n")[0]
        if key not in total_list_dict.keys():
            total_list_dict[key] = value
        else:
            sub_key = total_list_dict[key].split("_")[-1]
            total_list_dict[key] = {sub_key: total_list_dict[key]}
            sub_key = t.split("_")[-1].split("\n")[0]
            total_list_dict[key][sub_key] = value  ## 5512, 3190 // 5472, 5405

    for dir_ in tqdm(dir_list):
        key = dir_.split("_")[-2].split(os.sep)[-1]
        sub_folder_name = total_list_dict[key]
        if isinstance(sub_folder_name, dict):
            """
            LIDC-IDRI-0132, LIDC-IDRI-0132.20000101.3163.1
            LIDC-IDRI-0132, LIDC-IDRI-0132.20000101.5418.1
            LIDC-IDRI-0151, LIDC-IDRI-0151.20000101.3144.1
            LIDC-IDRI-0151, LIDC-IDRI-0151.20000101.5408.1
            LIDC-IDRI-0315, LIDC-IDRI-0315.20000101.5416.1
            LIDC-IDRI-0315, LIDC-IDRI-0315.20000101.5435.1
            LIDC-IDRI-0332, LIDC-IDRI-0332.20000101.30242.1
            LIDC-IDRI-0332, LIDC-IDRI-0332.20000101.9250.30078.1
            LIDC-IDRI-0355, LIDC-IDRI-0355.20000101.3190.1
            LIDC-IDRI-0355, LIDC-IDRI-0355.20000101.5512.1
            LIDC-IDRI-0365, LIDC-IDRI-0365.20000101.3175.1
            LIDC-IDRI-0365, LIDC-IDRI-0365.20000101.5409.1
            LIDC-IDRI-0442, LIDC-IDRI-0442.20000101.5405.1
            LIDC-IDRI-0442, LIDC-IDRI-0442.20000101.5472.1
            LIDC-IDRI-0484, LIDC-IDRI-0484.20000101.3100.2
            LIDC-IDRI-0484, LIDC-IDRI-0484.20000101.5417.1
            """
            sub_key = dir_.split("_")[-1]
            sub_folder_name = total_list_dict[key][sub_key]

        total_list.remove(f"{sub_folder_name}\n")
        save_subfolder_path = os.path.join(save_ct_path, sub_folder_name)
        if not os.path.exists(save_subfolder_path):
            os.makedirs(save_subfolder_path)  ## ct_xray_data.h5
        ct_path = glob.glob(os.path.join(dir_, '*.mha'))[0]
        _, ct_scan, _, _, _ = load_scan_mhda(ct_path)
        save_name = os.path.join(save_subfolder_path, "ct_xray_data.h5")
        f = h5py.File(save_name, 'w')
        f['ct'] = ct_scan
        f.close()
        xray_direction1 = cv2.imread(os.path.join(dir_, "xray1.png"))[..., 0]
        xray_direction2 = cv2.imread(os.path.join(dir_, "xray2.png"))[..., 0]

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
            os.path.join(save_xray_path, f"{sub_folder_name}_xray1.png"),
            xray_direction1,
        )
        cv2.imwrite(
            os.path.join(save_xray_path, f"{sub_folder_name}_xray2.png"),
            xray_direction2,
        )

    print(total_list)


if __name__ == "__main__":
    root_path = "D:/datasets"
    mda_path = "LIDC-IDRI-MDH_ctpro"
    save_path = "LIDC-HDF5-256_ct320"
    save_Xray_path = "LIDC-HDF5-256_ct320_plastimatch_xray"
    xray_size = 128
    make_h5_dataset_LIDC(root_path, mda_path, save_path, save_Xray_path, xray_size)
