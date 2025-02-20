# 环境依赖
glob, scipy, numpy, cv2, matplotlib, pydicom, SimpleITK, tqdm

# Usage Guidance

需要顺序执行以下几个程序。

## `0_dicom2mhd.py` : 将DICOM格式转为MHD格式

    调用参数：
    * Folders_Path : Dicom相对路径
    * save_path : MHD存储相对路径
    * root_path : 数据集根路径

关于Dicom源文件的目录结构示例见如下：

```
📦root_path/Folders_Path/
 ┣ 📂Dicom_0001
 ┃ ┣ 📜ct0001.dcm
 ┃ ┣ 📜ct0002.dcm
 ┃ ┣ 📜ct0003.dcm
 ┃ ┗ ...
 ┣ 📂Dicom_0002
 ┣ 📂Dicom_0003
 ┗ ...
```

## `1_ctpro_wo_mask.py` : 生成标准化处理的CT

    调用参数：
    * root_path : 数据集根路径
    * input_subdir : 第0步保存的MHD格式路径（即0步里面的save_path）
    * output_subdir : MHA保存相对路径
    * new_spacing : 重采样间距（一般不用改）
    * scale : 缩放大小
  
## `2_xraypro.py` : 调用plastimatch生成X光，并保存为PNG格式

在Plastimatch执行文件路径下运行该步 (ex. 'C:/Program Files/Plastimatch/bin')

    调用参数：
    * root_path : 数据集根路径
    * mda_path : MHA保存相对路径（即1步里面的output_subdir）
    * save_xray_path : X光保存路径
    * plasti_bin : Plastimatch的路径
    * Xray_size : 最终X光分辨率大小（和scale一致）
  
---

## 补充说明

### 关于Dicom

三维CT的DICOM数据通常是一个包含多张切片的序列，每张切片对应一个二维平面的DICOM文件。这些文件组合在一起就构成了完整的三维数据。文件结构可以分为以下几个部分：

1. **文件序列：**

    三维CT数据通常包含多个DICOM文件，每个文件表示一个特定的切片或层。
    切片的数量和顺序可能与CT扫描时的层厚度、间距以及所选的轴向（如轴状、冠状或矢状）有关。
    文件通常按照顺序命名，以确保重建三维体积时层与层之间的顺序正确。

2. **文件头（Metadata Header）：**

    每个DICOM文件的头部分包含丰富的元数据，这些信息描述了成像的具体细节和患者的信息。常见的元数据字段包括：

    + 患者信息：例如患者的姓名、年龄、性别等。
    + 扫描参数：如层厚度（Slice Thickness）、像素间距（Pixel Spacing）、扫描日期、曝光参数等。
    + 位置和方向：关键字段包括图像位置（Image Position Patient）和图像方向（Image Orientation Patient），这两个字段对构建三维模型很重要。它们描述了该切片在空间中的位置和方向。
    + Instance Number：通常用于标识切片在序列中的顺序。
    + 其他重要信息：设备信息、医院信息等。
  
3. **像素数据（Pixel Data）：**
   
   + 二维切片数据：DICOM文件的主要部分是图像的像素数据，通常存储在一个二维数组中。每张切片的像素矩阵大小可以是固定的，例如
   $256\times256$、$512\times512$等。
   + 像素值表示：每个像素的灰度值对应于一个CT值（也叫HU值），代表了物体密度，常用来区分组织类型。
   + 压缩方式：有些DICOM文件会使用无损或有损压缩算法，以减少存储空间需求。

   通过解析各个切片的图像位置和方向信息，可以将这些切片拼接成三维体积。

**如何处理这些DICOM文件**

读取三维DICOM数据时，需要按顺序读取所有切片文件，并根据其位置和顺序进行重建。``pydicom``库能够读取元数据，并提取出每张切片的像素数据。然后，借助诸如``SimpleITK``、``ITK``等库可以方便地处理和可视化三维DICOM数据。

---
### 关于``.mhd``和``.mha``文件

.mhd和.mha都是与医学图像数据相关的格式，都用于存储三维医学图像数据，特别是在使用ITK（Insight Segmentation and Registration Toolkit）等图像处理库时非常常见。两者的主要区别在于文件扩展名和一些细节实现，但功能和用途基本相同。将.mhd转换为.mha格式，通常并没有本质的区别，主要是在一些使用习惯或兼容性上有区别。

#### ``.mhd`` vs ``.mha``

> .mhd（MetaImage Header）：它是MetaImage格式的头文件，通常与一个二进制数据文件（.raw）配套使用。mhd文件本身包含了图像的元数据（如尺寸、分辨率、数据类型等），而实际的图像数据存储在.raw文件中。

> .mha（MetaImage）：它与.mhd类似，但它的图像数据直接嵌入到同一个文件中，不需要单独的二进制数据文件（如.raw）。.mha文件可以包含元数据以及实际的像素数据，因此它是一个单一文件格式，适用于那些希望将图像数据和元数据集中存储的场合。

---
