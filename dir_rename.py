import glob
import os

from natsort import natsorted


def rename_folders():
    # 设置文件夹路径
    folder_path = os.path.abspath(
        "D:/datasets/lidc_idri/LIDC-IDRI-MDH_ctpro_woMask"
    )  # 替换为实际文件夹路径

    # 遍历文件夹中的所有子文件夹
    for folder_name in os.listdir(folder_path):
        # 检查是否为文件夹
        old_folder_path = os.path.join(folder_path, folder_name)
        if os.path.isdir(old_folder_path):
            # 以"__"分割并取第二部分（A部分）
            A = folder_name.split('__')[1]
            # 以"-"分割并取最后一部分（B部分）
            B = folder_name.split('-')[-1]
            # 新的文件夹名称
            new_folder_name = f"{A}_{B}"
            new_folder_path = os.path.join(folder_path, new_folder_name)

            # 重命名文件夹
            os.rename(old_folder_path, new_folder_path)
            print(f"重命名 '{folder_name}' 为 '{new_folder_name}'")

    print("重命名完成！")


def sort_rename_files(source_dir_path, source_files_dir_path):
    map_dict = {}

    dir_list = natsorted(glob.glob(os.path.join(source_dir_path, '*')))
    for idx, dir_ in enumerate(dir_list):
        dir_name = dir_.split(os.sep)[-1]
        map_dict[dir_name] = idx

        new_folder_name = f"ct{idx:0>4}"
        new_folder_path = os.path.join(source_dir_path, new_folder_name)

        os.rename(dir_, new_folder_path)
    print("dir rename done!")

    files_list1 = natsorted(
        glob.glob(os.path.join(source_files_dir_path, '*_xray1.png'))
    )
    files_list2 = natsorted(
        glob.glob(os.path.join(source_files_dir_path, '*_xray2.png'))
    )

    for file1, file2 in zip(files_list1, files_list2):
        file1_name = file1.split(os.sep)[-1]
        file2_name = file2.split(os.sep)[-1]
        file1_key = file1_name.replace('_xray1.png', "")
        file2_key = file2_name.replace('_xray2.png', "")
        new_file1_name = f"{map_dict[file1_key]:0>4}_xray1.png"
        new_file2_name = f"{map_dict[file2_key]:0>4}_xray2.png"

        new_file1_path = os.path.join(source_files_dir_path, new_file1_name)
        new_file2_path = os.path.join(source_files_dir_path, new_file2_name)

        os.rename(file1, new_file1_path)
        os.rename(file2, new_file2_path)
    print("file rename done!")
