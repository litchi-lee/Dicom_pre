## To resize CT, Xray
import glob
import os
import sys

import h5py
import numpy as np
import parmap
from scipy.ndimage import zoom

sys.path.append(os.getcwd())
from tqdm import tqdm


# 检查是否为数组
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


#
def ct_preprocessing(dir_list, trg_ct_res, Save_path, CT_Save_path):
    base_ct_res = [320, 320, 320]
    file_name = "ct_xray_data.h5"

    # 遍历.h5文件目录
    for dir in tqdm(dir_list):
        file_path = os.path.join(dir, file_name)
        save_file_name = file_name

        sub_folder = os.path.basename(dir)
        ct_save_path = os.path.join(Save_path, sub_folder, "ct")

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

        os.makedirs(os.path.join(CT_Save_path, sub_folder), exist_ok=True)
        save_name = os.path.join(CT_Save_path, sub_folder, save_file_name)
        f = h5py.File(save_name, 'w')
        f['ct'] = ct
        f.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--Root_Source_path",
        type=str,
        help="Root_Source_path/*/ct_xray_data.h5",
        default="D:/datasets/lidc_idri/LIDC-HDF5-256_ct320",
    )
    # 指定并行处理的进程数
    parser.add_argument("--pm_processes", type=int, default=int(os.cpu_count() / 2))
    # 指定目标分辨率
    parser.add_argument("--target_res", type=int, default=128)
    args = parser.parse_args()
    Root_Source_path = os.path.abspath(args.Root_Source_path)
    print(f"processes: {args.pm_processes}")

    ## For multiple patients
    Save_path = os.path.abspath(f"{Root_Source_path}_ct{args.target_res}_CTSlice")
    CT_Save_path = os.path.abspath(f"{Root_Source_path}_ct{args.target_res}")
    dir_list = glob.glob(os.path.join(Root_Source_path, '*'))
    if args.pm_processes == 1:
        ct_preprocessing((dir_list, args.target_res, Save_path, CT_Save_path))
    else:
        data_splits = np.array_split(dir_list, args.pm_processes)
        parmap.starmap(
            ct_preprocessing,
            [
                (split, args.target_res, Save_path, CT_Save_path)
                for split in data_splits
            ],
            pm_pbar=True,
            pm_processes=args.pm_processes,
        )
