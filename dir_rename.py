import os

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
