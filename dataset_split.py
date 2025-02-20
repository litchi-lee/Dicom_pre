import os
import random

# 设置文件夹路径
folder_path = os.path.abspath("D:/datasets/lidc_idri/LIDC-IDRI-MDH_ctpro_woMask")
output_path = os.path.abspath(
    "D:/datasets/lidc_idri/dataset_list"
)  # 替换为保存txt文件的路径

# 获取所有子文件夹的名称列表
folders = [
    name
    for name in os.listdir(folder_path)
    if os.path.isdir(os.path.join(folder_path, name))
]

# 打乱文件夹顺序
random.shuffle(folders)

# 计算每个集合的数量
total_count = len(folders)
train_count = 781
val_count = 20
test_count = 82

# 划分数据集
train_set = folders[:train_count]
val_set = folders[train_count : train_count + val_count]
test_set = folders[train_count + val_count :]


# 保存为txt文件
def save_to_txt(file_list, file_name):
    with open(os.path.join(output_path, file_name), 'w') as f:
        for item in file_list:
            f.write(f"{item}\n")
    print(f"{file_name} 文件已生成，包含 {len(file_list)} 条数据")


# 保存训练集、验证集和测试集
save_to_txt(train_set, 'train.txt')
save_to_txt(val_set, 'val.txt')
save_to_txt(test_set, 'test.txt')

print("数据集划分完成！")
