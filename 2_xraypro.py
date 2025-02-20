import os
import time
from subprocess import check_output as qx

import cv2
import matplotlib.pyplot as plt
import numpy as np

try:
    from .ctxray_utils import load_scan_mhda, save_scan_mhda
except:
    from ctxray_utils import load_scan_mhda, save_scan_mhda

import os

# 需要修改的路径参数
root_path = "D:/datasets"
mda_path = "LIDC-IDRI-MDH_ctpro"
save_xray_path = "LIDC-IDRI-MDH_ctpro_xray"
plasti_bin = "D:/programs/Plastimatch/bin"
Xray_size = 320

root_dir = os.path.abspath(root_path)
root_path = os.path.join(root_dir, mda_path)
save_root_path = os.path.join(root_dir, save_xray_path)
plasti_path = os.path.abspath(plasti_bin)


# 计算图像中心点的世界坐标
def get_center(origin, size, spacing):
    origin = np.array(origin)
    size = np.array(size)
    spacing = np.array(spacing)
    center = origin + (size - 1) / 2 * spacing
    return center


# 将numpy数组转换为字符串
def array2string(ndarray):
    ret = ""
    for i in ndarray:
        ret = ret + str(i) + " "
    return ret[:-2]


# 将PFM文件转换为PNG文件
def savepng(filename, direction):
    # 读取PFM文件
    raw_data = cv2.imread(filename, cv2.IMREAD_UNCHANGED)
    max_value = raw_data.max()

    # 将像素值缩放到0-255之间
    im = (raw_data / max_value * 255).astype(np.uint8)

    # 翻转图像
    if direction == 1:
        im = np.fliplr(im)

    savedir, _ = os.path.split(filename)
    outfile = os.path.join(savedir, "xray{}.png".format(direction))

    # 保存PNG文件
    plt.imsave(outfile, im, cmap=plt.cm.gray)
    image = cv2.imread(outfile)
    gray = cv2.split(image)[0]
    cv2.imwrite(outfile, gray)


# 生成虚拟X光图像
def make_input():
    files_list = os.listdir(root_path)

    start = time.time()
    for fileIndex, fileName in enumerate(files_list):
        t0 = time.time()
        print('Begin {}/{}: {}'.format(fileIndex + 1, len(files_list), fileName))

        # 创建保存目录
        saveDir = os.path.join(save_root_path, fileName)
        if not os.path.exists(saveDir):
            os.makedirs(saveDir)

        # 加载原始CT扫描.mha文件
        try:
            savePath = os.path.join(
                os.path.join(root_path, fileName), '{}.mha'.format(fileName)
            )
            ct_itk, ct_scan, ori_origin, ori_size, ori_spacing = load_scan_mhda(
                savePath
            )
        except:
            savePath = os.path.join(os.path.join(root_path, fileName), 'ct_file.mha')
            ct_itk, ct_scan, ori_origin, ori_size, ori_spacing = load_scan_mhda(
                savePath
            )

        # 计算图像中心点的世界坐标
        center = get_center(ori_origin, ori_size, ori_spacing)

        # 调用Plastimatch生成虚拟X光图像
        cmd_str = f'{os.path.join(plasti_path, "plastimatch.exe")} adjust --input "{savePath}" --output "{saveDir}/out.mha" --pw-linear "0, -1000"'
        output = qx(cmd_str)

        # 生成两个方向的虚拟X光图像
        directions = [1, 2]
        for i in directions:
            if i == 1:
                nrm = "0 1 0"
            else:
                nrm = "1 0 0"
            '''
            plastimatch usage
            -t : save format
            -g : sid sad [DistanceSourceToPatient]:541 
                         [DistanceSourceToDetector]:949.075012
            -r : output image resolution
            -o : isocenter position
            -z : physical size of imager
            -I : input file in .mha format
            -O : output prefix
            '''
            # 调用Plastimatch的drr.exe生成虚拟X光图像
            # 其中-g "700 949"是根据具体情况设置的，SID表示源到物体的距离，SAD表示源到探测器的距离
            # -r "320 320"表示输出图像的分辨率
            cmd_str = f'{os.path.join(plasti_path, "drr.exe")} -t pfm -nrm "{nrm}" -g "700 949" -r "{Xray_size} {Xray_size}" -o "{array2string(center)}" -z "500 500" -I "{saveDir}/out.mha" -O "{saveDir}/{i}"'
            output = qx(cmd_str)
            pfmFile = saveDir + '/{}'.format(i) + '0000.pfm'
            savepng(pfmFile, i)

        # 删除生成的中间文件
        os.remove(saveDir + '/out.mha')
        t1 = time.time()
        print('End! Case time: {}'.format(t1 - t0))

    end = time.time()
    print('Finally! Total time: {}'.format(end - start))


if __name__ == '__main__':
    make_input()
