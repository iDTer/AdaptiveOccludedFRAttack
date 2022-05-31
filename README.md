# AdaptiveOccludedFRAttack


### Requirements

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

### Dataset

使用Labelme标注的口罩分割数据集——Seg_data：https://www.mirrored.to/files/1BH1JMJ4/Seg_data.zip_links

下载口罩分割数据集解压并划分train和val，放置于目录`AOA2/Mask_Seg/data`

裁剪后的CelebA人脸数据集：https://www.mirrored.to/files/0UWEVPAS/img_align_celeba_croped.zip_links

基于CelebA生成的口罩遮挡人脸数据集：https://www.mirrored.to/files/4EJ0XQW3/img_align_celeba_croped_masked.zip_links

AOA1 Mask：https://www.mirrored.to/files/HOLZLO0S/mask.zip_links

model：...

## Training

#### 1. AOA1

* STEP 1. 运行下面的命令，计算人脸图像之间的相似度，结果输出为likehood.json

```
python AOA1/calc_likehood.py
```

- STEP 2. 生成遮挡人脸图像的干扰限制区域，输出目录为mask：

```
python pretreatment/prepro.py --vocab_size 8000
```

* STEP 3. 生成基于局部特征特征增强口罩遮挡人脸识别的对抗样本

```
python target_attack
```

* STEP 4. 攻击效果验证，验证以百度FR为例，需要自行注册百度SDK，完善以下信息

```
APP_ID = ''
API_KEY = ''
SECRET_KEY = ''
```

运行下面的命令，构建在线的人脸数据库，

```
python face_register.py
```

#### 2. AOA2





## Result

当输入：

```

```

输出句子：





