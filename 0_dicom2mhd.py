import glob
import os

import numpy as np
import pydicom as dc
import SimpleITK as sitk
from tqdm import tqdm


def dicom_to_mhd(root_dir, folders_path, save_path, text_file_name="dicom2mhd.txt"):
    Folders_Path = os.path.join(root_dir, folders_path)
    print(f"Folders_Path: {Folders_Path}")
    save_path = os.path.join(root_dir, save_path)

    text_path = os.path.join(os.path.dirname(save_path), text_file_name)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(text_path, 'w') as file:
        file.write(
            "filename\tRescaleSlope\tRescaleIntercept\tDimension\tSpacing\tOrigin\n"
        )

    print(f"Using glob pattern: {os.path.join(Folders_Path, '*')}")
    dir_list = glob.glob(os.path.join(Folders_Path, '*'))
    print(f"提取的dicom示例(0号): {dir_list[0]}")
    for dir_ in tqdm(dir_list):
        files = glob.glob(os.path.join(dir_, '*.dcm'))
        if len(files) > 10:
            # 目录和文件名处理
            dir_names = dir_[len(Folders_Path) :].split(os.sep)
            save_fileName = "__".join(dir_names[:-1])

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

            # 初始化一个与dicom文件相同大小的数组
            NpArrDc = np.zeros(Dimension, dtype=ReferenceImage.pixel_array.dtype)

            sliceIDs = []
            for filename in DICOM_LIST:
                image = dc.read_file(filename)
                sliceID = image.data_element("InstanceNumber").value - 1
                sliceIDs.append(sliceID)

            sliceIDs.sort()
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

                # 将图像像素值整体加上1024，使其在0-2048之间
                image_pixel_array += 1024
                NpArrDc[:, :, sliceID] = image_pixel_array
            with open(text_path, 'a') as file:
                file.write(
                    f"{save_fileName}\t{rescale_slope}\t{rescale_intercept}\t{dimension}\t{spacing}\t{origin}\n"
                )

            NpArrDc = np.transpose(NpArrDc, (2, 0, 1))
            sitk_img = sitk.GetImageFromArray(NpArrDc, isVector=False)
            sitk_img.SetSpacing(Spacing)
            sitk_img.SetOrigin(Origin)
            sitk.WriteImage(sitk_img, os.path.join(save_path, save_fileName + ".mhd"))
    print("dicom2mhd done!")


if __name__ == "__main__":
    # 示例调用（需要修改root_dir、folders_path、save_path）
    root_dir = "D:/datasets"
    folders_path = "lidc_idri/manifest-1730897841304/LIDC-IDRI"
    save_path = "lidc_idri/LIDC-IDRI-MDH"
    dicom_to_mhd(root_dir, folders_path, save_path)
