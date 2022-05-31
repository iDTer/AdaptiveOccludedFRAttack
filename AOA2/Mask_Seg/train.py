from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
import os
from net import simpleNet5
from dataset import SegDataset
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# Visualization
writer = SummaryWriter()

batchsize = 16
epochs = 200
imagesize = 256
cropsize = 224
train_data_path = './data/train'
val_data_path = './data/val'

# 数据预处理
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

"""图像分割数据集"""
train_dataset = SegDataset(train_data_path, imagesize, cropsize, data_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True)
val_dataset = SegDataset(val_data_path, imagesize, cropsize, data_transform)
val_dataloader = DataLoader(val_dataset, batch_size=val_dataset.__len__(), shuffle=True)

image_datasets = {'train': train_dataset, 'val': val_dataset}
dataloaders = {'train': train_dataloader, 'val': val_dataloader}

"""定义网络，优化目标，优化方法"""
device = torch.device('cuda')
net = simpleNet5().to(device)
# 使用softmax loss损失，输入label是图片
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=1e-1, momentum=0.9)
# 每50个epoch，学习率衰减
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

if not os.path.exists('checkpoints'):
    os.mkdir('checkpoints')

for epoch in range(1, epochs+1):
    print('Epoch {}/{}'.format(epoch, epochs-1))
    for phase in ['train', 'val']:
        if phase == 'train':
            scheduler.step()
            # Set model to training mode
            net.train(True)
        else:
            # Set model to evaluate mode
            net.train(False)

        running_loss = 0.0
        running_accs = 0.0

        n = 0
        for data in dataloaders[phase]:
            imgs, labels = data
            img, label = imgs.to(device).float(), labels.to(device).float()
            output = net(img)
            # 得到损失
            loss = criterion(output, label.long())

            output_mask = output.cpu().data.numpy().copy()
            output_mask = np.argmax(output_mask, axis=1)
            y_mask = label.cpu().data.numpy().copy()
            # 计算精度
            acc = (output_mask == y_mask)
            acc = acc.mean()

            optimizer.zero_grad()
            if phase == 'train':
                # 梯度置零，反向传播，参数更新
                loss.backward()
                optimizer.step()

            running_loss += loss.data.item()
            running_accs += acc
            n += 1

        epoch_loss = running_loss / n
        epoch_acc = running_accs / n

        if phase == 'train':
            writer.add_scalar('data/trainloss', epoch_loss, epoch)
            writer.add_scalar('data/trainacc', epoch_acc, epoch)
            print('train epoch_{} loss={}'.format(epoch, epoch_loss))
            print('train epoch_{} acc={}'.format(epoch, epoch_acc))
        else:
            writer.add_scalar('data/valloss', epoch_loss, epoch)
            writer.add_scalar('data/valacc', epoch_acc, epoch)
            print('val epoch_{} loss={}'.format(epoch, epoch_loss))
            print('val epoch_{} acc={}'.format(epoch, epoch_acc))

    if epoch % 100 == 0:
        torch.save(net, 'checkpoints/model_epoch_{}.pt'.format(epoch))
        print('checkpoints/model_epoch_{}.pt saved!'.format(epoch))

writer.export_scalars_to_json('./all_scalars.json')
writer.close()



