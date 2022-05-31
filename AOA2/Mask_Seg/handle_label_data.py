import os
import cv2
import numpy as np


dataPath = "./data/train"
for root, dirs, files in os.walk(dataPath):
    if root != dataPath:
        if not files:
            continue
        else:
            label_path = os.path.join(root, 'label.png')
            label = cv2.imread(label_path, 0)
            # pix = np.unique(label)
            # print(pix)
            # print(type(label))
            label[label == 38] = 1
            # print(label)
            new_label_path = os.path.join(root, 'instance_label.png')
            cv2.imwrite(new_label_path, label)



