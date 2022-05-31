from skimage import io
import os
import dlib
import numpy as np

predictor_path = "models/dlib_models/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)


def get_landmarks(img):
    dets = detector(img, 1)
    landmarks = np.zeros((17, 2))
    for k, d in enumerate(dets):
        shape = predictor(img, d)
        landmarks[0] = (shape.part(48).x, shape.part(48).y)
        for i in range(6):
            landmarks[1 + i] = (shape.part(59 - i).x, shape.part(59 - i).y)
        for i in range(10):
            landmarks[7 + i] = (shape.part(26 - i).x, shape.part(26 - i).y)
    return landmarks


def mask_landmarks(img):
    dets = detector(img, 1)
    landmarks = np.zeros((11, 2))
    for k, d in enumerate((dets)):
        shape = predictor(img, d)
        landmarks[0] = (shape.part(30).x, shape.part(30).y)
        for i in range(10):
            landmarks[1 + i] = (shape.part(26 - i).x, shape.part(26 - i).y)
    return landmarks


def inside(X, Y, Region):
    j = len(Region) - 1
    flag = False
    for i in range(len(Region)):
        if Region[i][1] < Y <= Region[j][1] or Region[j][1] < Y <= Region[i][1]:
            if Region[i][0] + (Y - Region[i][1]) / (Region[j][1] - Region[i][1]) * (Region[j][0] - Region[i][0]) < X:
                flag = not flag
        j = i
    return flag


paths = []
picpath = './img_align_celeba_croped'
print(picpath)
dire = None
for root, dirs, files in os.walk(picpath):
    for f in files:
        paths.append(os.path.join(root, f))

# num = 1
print("processing image  =========>")
for path in paths:
    img = io.imread(path)
    region = mask_landmarks(img)
    shape = list(img.shape) + [3]
    img1 = img.copy()
    for i in range(shape[0]):
        for j in range(shape[1]):
            if not inside(j, i, region):
                img1[i, j] = (0, 0, 0)
            else:
                img1[i, j] = (255, 255, 255)
    to_save = os.path.join('./mask', path.split('\\')[-1])
    io.imsave(to_save, img1)
    # num += 1


