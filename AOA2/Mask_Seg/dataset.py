import os
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import numpy as np
import random
from PIL import Image


class SegDataset(Dataset):
    def __init__(self, dataPath, imagesize, cropsize, transform=None):
        self.samples = []
        self.imagesize = imagesize
        self.cropsize = cropsize
        self.transfrom = transform
        if self.transfrom is None:
            self.transfrom = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])

        for root, dirs, files in os.walk(dataPath):
            if root != dataPath:
                image_path = os.path.join(root, 'img.png')
                label_path = os.path.join(root, 'instance_label.png')
                self.samples.append((image_path, label_path))

    def __getitem__(self, index):
        imagepath, labelpath = self.samples[index]
        image = cv2.imread(imagepath)
        label = cv2.imread(labelpath, 0)

        # img = Image.open(labelpath)
        # img = np.array(img)
        # print(img.shape)

        # 添加基本的数据增强，对图片和标签保持一致
        # 添加固定尺度的随机裁剪，使用最近邻缩放（不产生新的灰度值）+裁剪
        image = cv2.resize(image, (self.imagesize, self.imagesize), interpolation=cv2.INTER_NEAREST)
        label = cv2.resize(label, (self.imagesize, self.imagesize), interpolation=cv2.INTER_NEAREST)
        offsetx = np.random.randint(self.imagesize - self.cropsize)
        offsety = np.random.randint(self.imagesize - self.cropsize)
        image = image[offsety: offsety+self.cropsize, offsetx: offsetx+self.cropsize]
        label = label[offsety: offsety+self.cropsize, offsetx: offsetx+self.cropsize]

        # 只对image做预处理
        return self.transfrom(image), label

    # 统计数据集大小
    def __len__(self):
        return len(self.samples)


if __name__ == '__main__':
    path_to_data = "./data/train"
    imagesize = 256
    cropsize = 224
    mydataset = SegDataset(path_to_data, imagesize, cropsize)
    # print(mydataset.samples)
    # print(mydataset.__len__())

    imagepath, labelpath = mydataset.samples[1]
    image, label = mydataset.__getitem__(2)
    # print("image", image)
    # print("label", label)
    # srcimage = cv2.imread(open(filetxt, 'r').readlines()[0].strip().split(' ')[0])
    srcimage = cv2.imread(imagepath)
    # cv2.namedWindow("image", 0)
    # cv2.imshow("srcimage", srcimage)
    # cv2.imshow("label", label)
    # cv2.namedWindow("cropimage", 0)
    # cv2.imshow("cropimage", ((image.numpy() * 0.5 + 0.5) * 255).astype(np.uint8).transpose(1, 2, 0))
    # cv2.waitKey(0)

    print(image.shape)
    print(label.shape)
