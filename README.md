# AdaptiveOccludedFRAttack


### Requirements

python==3.6
dlib==19.22.1
numpy==1.19.4
opencv_python==4.5.2.54
Pillow==9.1.1
requests==2.25.1
scikit_image==0.17.2
scipy==1.5.4
skimage==0.0
tensorboardX==2.5
torch==1.8.0+cu101
torchvision==0.9.0+cu101

### Dataset

使用Labelme标注的口罩分割数据集——Seg_data：https://www.mirrored.to/files/1BH1JMJ4/Seg_data.zip_links

下载口罩分割数据集解压并划分train和val，放置于目录`AOA2\Mask_Seg\data`

裁剪后的CelebA人脸数据集：https://www.mirrored.to/files/0UWEVPAS/img_align_celeba_croped.zip_links

基于CelebA生成的口罩遮挡人脸数据集：https://www.mirrored.to/files/4EJ0XQW3/img_align_celeba_croped_masked.zip_links



## Training

* STEP 1. 运行下面的命令，生成

```
python pretreatment/prepro.py
```

如果你想调整默认的词典大小(default:32000)，可以进行下面的命令：

```
python pretreatment/prepro.py --vocab_size 8000
```

它会创建两个文件 `barrages_data/prepro` and `barrages_data/segmented`.

* STEP 2. 训练模型

```
python train.py
```

参数设置放在 `hparams.py` ，可以根据里面的参数进行对应设置，比如：

```
python train.py --logdir myLog --batch_size 256 --dropout_rate 0.5
```

* STEP 3. 根据输入的句子，生成弹幕

```
python barrrages_generate.py
```

## Result

当输入：

```

```

输出句子：





