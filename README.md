# AdaptiveOccludedFRAttack


## Requirements

- python==3.6
- dlib==19.22.1
- numpy==1.19.4
- opencv_python==4.5.2.54
- Pillow==9.1.1
- requests==2.25.1
- scikit_image==0.17.2
- scipy==1.5.4
- tensorboardX==2.5
- torch==1.8.0+cu101
- torchvision==0.9.0+cu101

## Dataset

AOA1 Mask: https://www.mirrored.to/files/HOLZLO0S/mask.zip_links

Model: https://url65.ctfile.com/f/37075665-587377309-3e8bca (key: 6136)

Labelme[1] labeled masks which used in mask segmentation——Seg_data: https://www.mirrored.to/files/1BH1JMJ4/Seg_data.zip_links. Download and unpack the zip, and divide train and val into `AOA2/Mask_Seg/data`

Cropped CelebA[2] face dataset: https://www.mirrored.to/files/0UWEVPAS/img_align_celeba_croped.zip_links

Mask face dataset based on CelebA: https://www.mirrored.to/files/4EJ0XQW3/img_align_celeba_croped_masked.zip_links

## Training

### 1. AOA1

* STEP 1. 运行下面的命令，计算人脸图像之间的相似度，结果输出为likehood.json

```
python AOA1/calc_likehood.py
```

- STEP 2. 生成遮挡人脸图像的干扰限制区域，输出目录为mask：

```
python AOA1/generate_mask.py
```

* STEP 3. 生成基于局部特征特征增强口罩遮挡人脸识别的对抗样本

```
python AOA1/target_attack
```

* STEP 4. 攻击效果验证，以百度FR为例，需要自行注册百度SDK，完善以下信息

```
APP_ID = ''
API_KEY = ''
SECRET_KEY = ''
```

运行下面的命令，构建在线的人脸数据库，

```
python AOA1/baiduface/face_register.py
```

判断攻击是否成功

```
python AOA1/baiduface/face_rec.py
```

### 2. AOA2

- STEP 1. 运行下面的命令，训练口罩遮挡人脸分割模型，

```
python AOA2/Mask_Seg/train.py
```

- STEP 2. 生成基于人脸修复口罩遮挡人脸识别的对抗样本

```
python AOA2/src/AOA2.py
```
## Reference

[1] https://github.com/wkentaro/labelme

[2] Liu Z, Luo P, Wang X, et al. Deep learning face attributes in the wild[C]//Proceedings of the IEEE international conference on computer vision. 2015: 3730-3738.



