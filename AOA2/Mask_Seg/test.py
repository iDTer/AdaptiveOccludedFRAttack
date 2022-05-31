from unittest import result
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import time
import os
import cv2
import sys
import torch.nn.functional as F
import numpy as np


data_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

modelpath = './old_model/model_epoch_200.pt' # 模型目录
net = torch.load(modelpath, map_location='cpu')
net.eval()  # 设置为推理模式，不会更新模型的k，b参数

testpath = './data/test'
imagepaths = os.listdir(testpath)  # 测试图片目录
torch.no_grad()  # 停止autograd模块的工作，加速和节省显存

for imagepath in imagepaths:
    image = cv2.imread(os.path.join(testpath, imagepath)) # 读取图像
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_NEAREST)
    imagblob = data_transforms(image).unsqueeze(0) # 填充维度，从3维到4维
    predict = F.softmax(net(imagblob), dim=1).cpu().data.numpy().copy() # 获得原始网络输出，多通道
    predict = np.argmax(predict, axis=1) # 得到单通道的label
    result = np.squeeze(predict) # 降低维度，从4维到3维
    # print(np.max(result))
    result = (result*127).astype(np.uint8) # 灰度拉伸，方便可视化

    resultimage = image.copy()
    for y in range(0, result.shape[0]):
        for x in range(0, result.shape[1]):
            if result[y][x] == 127:
                resultimage[y][x] = (0, 0, 255)
            elif result[y][x] == 254:
                resultimage[y][x] = (0, 255, 255)
    combine_result = np.concatenate([image, resultimage], axis=1)
    cv2.imwrite(os.path.join('./data/results', imagepath), combine_result)
