import json
import random
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import os
import sys
from PIL import Image
import torchvision
import warnings
import multiprocessing
import torch.multiprocessing
from model_irse import IR_50, IR_101, IR_152

warnings.filterwarnings("ignore")
to_torch_tensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                  torchvision.transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

break_threshold = -0.25
learning_rate = 8
momentum = 0.5
mask_path = 'mask'


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""
    import scipy.stats as st

    interval = (2 * nsig + 1.) / kernlen
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


class GaussianBlur(nn.Module):
    def __init__(self, kernel):
        super(GaussianBlur, self).__init__()
        self.kernel_size = len(kernel)
        print('kernel size is {0}.'.format(self.kernel_size))
        assert self.kernel_size % 2 == 1, 'kernel size must be odd.'

        self.kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        self.weight = nn.Parameter(data=self.kernel, requires_grad=False)

    def forward(self, x):
        x1 = x[:, 0, :, :].unsqueeze(1)
        x2 = x[:, 1, :, :].unsqueeze(1)
        x3 = x[:, 2, :, :].unsqueeze(1)
        padding = self.kernel_size // 2
        x1 = F.conv2d(x1, self.weight, padding=padding)
        x2 = F.conv2d(x2, self.weight, padding=padding)
        x3 = F.conv2d(x3, self.weight, padding=padding)
        x = torch.cat([x1, x2, x3], dim=1)
        return x


def get_gaussian_blur(kernel_size, device):
    kernel = gkern(kernel_size, 2).astype(np.float32)
    gaussian_blur = GaussianBlur(kernel)
    return gaussian_blur.to(device)


def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = torch.div(input, norm)
    return output


# tv_loss
def tv_loss(input_t):
    temp1 = torch.cat((input_t[:, :, 1:, :], input_t[:, :, -1, :].unsqueeze(2)), 2)
    temp2 = torch.cat((input_t[:, :, :, 1:], input_t[:, :, :, -1].unsqueeze(3)), 3)
    temp = (input_t - temp1) ** 2 + (input_t - temp2) ** 2
    return temp.sum()


# Return the initialization noise of xavier_normal
def get_init_noise(device):
    noise = torch.Tensor(1, 3, 112, 112)
    noise = torch.nn.init.xavier_normal(noise, gain=1)
    return noise.to(device)


# Return the model (backbone) of face recognition, which outputs a 512-dimensional vector
def get_model(model, param, device, proportion, kind):
    if kind != '-1,1' and kind != '0,1':
        raise 'no this kind model!'
    m = model([112, 112])
    m.eval()
    m.to(device)
    m.load_state_dict(torch.load(param, map_location=device))
    model_dict = {'model': m, 'proportion': proportion, 'kind': kind}
    return model_dict


# Return the model pool
def get_model_pool(device):
    model_pool = [get_model(IR_50, 'models/backbone_ir50_ms1m_epoch120.pth', device, 2, '-1,1'),
                  get_model(IR_50, 'models/Backbone_IR_50_LFW.pth', device, 1, '-1,1'),
                  get_model(IR_101, 'models/Backbone_IR_101_Batch_108320.pth', device, 1, '-1,1'),
                  get_model(IR_152, 'models/Backbone_IR_152_MS1M_Epoch_112.pth', device, 1, '-1,1')]
    return model_pool


# Randomly choose a model
def random_choose_model(model_pool):
    s = len(model_pool)
    index = random.randint(0, s - 1)
    return model_pool[index]


# Normalization
def normal_model_proportion(model_pool):
    sum1 = 0
    for model_dict in model_pool:
        sum1 += model_dict['proportion']
    for model_dict in model_pool:
        model_dict['proportion'] /= sum1
    return model_pool


# Return the torch.tensor image pool
def get_img_pool(person_list, device):
    person_pool = []
    for el in person_list:
        person_pool.append(to_torch_tensor(Image.open(el)).unsqueeze_(0).to(device))
    return person_pool


# Single step iteration
def iter_step(tmp_noise, origin_img, target_img, mask, gaussian_blur, model_pool, index, loss1_v, momentum=0, lr=1):
    tmp_noise.requires_grad = True
    noise = gaussian_blur(tmp_noise)
    noise *= mask
    # print('loss1_v = {}, momentum = {}'.format(loss1_v, momentum))
    # print('last loss1_v is ' + str(loss1_v))
    loss1 = 0
    score1 = 0
    score2 = 0
    for model_dict in model_pool:
        model = model_dict['model']
        proportion = model_dict['proportion']
        v1 = l2_norm(model(origin_img + noise))
        v2_1 = l2_norm(model(origin_img)).detach_()
        v2_2 = l2_norm(model(target_img)).detach_()
        tmp1 = (v1 * v2_1).sum()
        tmp2 = (v1 * v2_2).sum()
        # print(tmp1.item(), tmp2.item())
        r1 = 1
        r2 = 1
        if tmp1 < 0.3:
            r1 = 0
        if tmp2 > 0.7:
            r2 = 0
        loss1 += (r1 * tmp1 - r2 * tmp2) * proportion
        score1 += tmp1.item() * proportion
        score2 += tmp2.item() * proportion

    loss1.backward(retain_graph=True)
    loss1_v = tmp_noise.grad.detach() * (1 - momentum) + loss1_v * momentum
    tmp_noise.grad.data.zero_()
    print('loss1_v.abs().sum() is {}'.format(loss1_v.abs().sum()))

    r3 = 1
    if index > 100:
        r3 *= 0.1
    if index > 200:
        r3 *= 0.1

    loss2 = (noise ** 2).sum().sqrt()
    loss3 = tv_loss(noise)
    loss = r3 * 0.025 * loss2 + r3 * 0.004 * loss3
    loss.backward()
    # print(tmp_noise.grad.detach().abs().sum())

    print('score1 is {}, score2 is {}, loss2.item() is {}'.format(score1, score2, loss2.item() * 128 / 112 / 1.732))

    tmp_noise = tmp_noise.detach() - lr * (tmp_noise.grad.detach() + loss1_v)
    tmp_noise = (tmp_noise + origin_img).clamp_(-1, 1) - origin_img
    tmp_noise = tmp_noise.clamp_(-0.2, 0.2)
    return tmp_noise, score1, score2, loss1_v


# Multi-step iteration, call the above single-step iterative function
def noise_iter(model_pool, origin_img, target_img, mask, gaussian_blur, device):
    tmp_noise = get_init_noise(device)
    index = 0
    loss1_v = 0
    while True:
        index += 1
        tmp_noise, socre1, socre2, loss1_v = iter_step(tmp_noise, origin_img, target_img, mask, gaussian_blur,
                                                       model_pool, index, loss1_v, momentum, lr=learning_rate)
        yield tmp_noise, socre1, socre2


# compute an adversarial example
def cal_adv(origin_name, target_name, mask_name, model_pool, gaussian_blur, device):
    origin_img = to_torch_tensor(Image.open(origin_name))
    origin_img = origin_img.unsqueeze_(0).to(device)
    target_img = to_torch_tensor(Image.open(target_name))
    target_img = target_img.unsqueeze_(0).to(device)
    mask = torchvision.transforms.ToTensor()(Image.open(mask_name))
    mask = mask.unsqueeze_(0).to(device)

    generator = noise_iter(model_pool, origin_img, target_img, mask, gaussian_blur, device)

    scores = 0
    i = 0
    scores1 = 0
    while True:
        tmp_noise, socre1, socre2 = next(generator)
        socre = socre1 - socre2
        if socre < -0.4:
            socre = -0.4
        scores = 0.5 * socre + 0.5 * scores
        i += 1
        if i > 300:
            f = open('hard.txt', 'a')
            f.write(origin_name + ';' + target_name + ';' + str(scores) + '\n')
            f.close()
            _, o_f = os.path.split(origin_name)
            _, t_f = os.path.split(target_name)
            print('origin img is %s, target img is %s, iter %d, socre is %0.3f'
                  % (o_f, t_f, i, scores))
            break

        if scores < break_threshold:
            _, o_f = os.path.split(origin_name)
            _, t_f = os.path.split(target_name)
            print('origin img is %s, target img is %s, iter %d, socre is %0.3f'
                  % (o_f, t_f, i, scores))
            break

    return gaussian_blur(tmp_noise) * mask, origin_img, i


# employ one process to generate a adversarial sample
def one_process_run(people_list, model_pool, device):
    f = open("likelihood.json")
    likelihood = json.load(f)
    gaussian_blur = get_gaussian_blur(kernel_size=3, device=device)
    l2 = []
    for origin_name in people_list:
        # Generate a adversarial sample
        noise, _, iter_num = cal_adv(os.path.join('img_align_celeba_croped_masked', origin_name),
                                     os.path.join('img_align_celeba_croped_masked', likelihood[origin_name][0]),
                                     os.path.join(mask_path, origin_name), model_pool, gaussian_blur, device)

        # Remove small disturbances
        THRESHOLD = 2 + 0.00001
        noise = torch.round(noise * 127.5)[0].cpu().numpy()
        noise = noise.swapaxes(0, 1).swapaxes(1, 2)
        noise = noise.clip(-25.5, 25.5)
        noise = np.where((noise > -THRESHOLD) & (noise < THRESHOLD), 0, noise)
        origin_img = np.array(Image.open(os.path.join('img_align_celeba_croped_masked', origin_name)), dtype=float)
        numpy_adv_sample = (origin_img + noise).clip(0, 255)
        adv_sample = Image.fromarray(np.uint8(numpy_adv_sample))

        # Enumerate L2 norm
        noise = (numpy_adv_sample - origin_img)
        noise_l2_norm = np.sqrt(np.sum(noise ** 2, axis=2)).mean()
        l2.append(noise_l2_norm)
        print('%s noise l2_norm is %.4f' % (origin_name, noise_l2_norm))

        f = open('iter_num.txt', 'a')
        f.write('%s %d %.4f\n' % (origin_name, iter_num, noise_l2_norm))
        f.close()

        # Save image
        if os.path.exists('images/') is False:
            os.mkdir('images/')
        jpg_img = 'images/' + origin_name
        png_img = jpg_img.replace('jpg', 'png')
        adv_sample.save(png_img)
        os.rename(png_img, jpg_img)
    return l2


def run(p_pool, people_list, device):
    double_process = True
    model_pool = get_model_pool(device)
    model_pool = normal_model_proportion(model_pool)
    print('----model load over----')
    res = []
    if double_process:
        for model_dict in model_pool:
            model_dict['model'].share_memory()
        res.append(p_pool.apply_async(one_process_run, args=(people_list[:len(people_list) // 2], model_pool, device)))
        res.append(p_pool.apply_async(one_process_run, args=(people_list[len(people_list) // 2:], model_pool, device)))
    else:
        res.append(p_pool.apply_async(one_process_run, args=(people_list, model_pool, device)))
    return res


def main():
    faces = os.listdir('img_align_celeba_croped_masked')
    faces.sort(key=lambda x: int(x[:-4]))
    # Set device
    device = torch.device('cuda:0')
    model_pool = get_model_pool(device)
    model_pool = normal_model_proportion(model_pool)
    run(faces, model_pool, device)


if __name__ == '__main__':
    main()
