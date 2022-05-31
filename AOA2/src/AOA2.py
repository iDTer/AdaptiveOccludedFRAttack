import copy
import random
import torch
from torch.autograd import Variable
# In-repo imports
from util import (save_prediction_image,
                  save_input_image,
                  save_image_difference,
                  calculate_mask_similarity,
                  calculate_image_distance)
from mask_dataset import MaskDatasetTest
from util import load_model

random.seed(4)


def update_perturbation_multiplier(beta, tau, iou):
    return beta * iou + tau


def calculate_l2_loss(x, y):
    loss = (x - y) ** 2
    for a in reversed(range(1, loss.dim())):
        loss = loss.sum(a, keepdim=False)
    loss = loss.sum()
    return loss


class AdaptiveOccludedFRAttack:
    def __init__(self, device_id, model, tau, beta):
        self.device_id = device_id
        self.model = model
        self.tau = tau
        self.beta = beta

    def calculate_pred_loss(self, target_mask, pred_out, model_output):
        loss = 0
        for single_class in self.unique_classes:
            out_channel = model_output[0][single_class]
            optimization_mask = copy.deepcopy(target_mask)
            optimization_mask[optimization_mask != single_class] = self.temporary_class_id
            optimization_mask[optimization_mask == single_class] = 1
            optimization_mask[optimization_mask == self.temporary_class_id] = 0
            prediction_mask = copy.deepcopy(pred_out)[0]
            prediction_mask[prediction_mask != single_class] = self.temporary_class_id
            prediction_mask[prediction_mask == single_class] = 0
            prediction_mask[prediction_mask == self.temporary_class_id] = 1
            channel_loss = torch.sum(out_channel * optimization_mask * prediction_mask)
            loss = loss + channel_loss
        return loss

    def perform_attack(self, input_image, org_mask, target_mask, unique_class_list,
                       total_iter=501, save_samples=True, save_path='../res/', verbose=True):
        if save_samples:
            save_prediction_image(org_mask.numpy(), 'original_mask', save_path)
            save_prediction_image(target_mask.numpy(), 'target_mask', save_path)
        f = open('./pert_mul.txt', 'a')
        # Unique classes are needed to simplify prediction loss
        self.unique_classes = unique_class_list
        # Have a look at calculate_pred_loss to see where this is used
        self.temporary_class_id = random.randint(0, 999)
        while self.temporary_class_id in self.unique_classes:
            self.temporary_class_id = random.randint(0, 999)

        # Assume there is no overlapping part for the first iteration
        pert_mul = update_perturbation_multiplier(self.beta, self.tau, 0)
        # Get a copy of target mask to use it for stats
        target_mask_numpy = copy.deepcopy(target_mask).numpy()
        # Target mask
        target_mask = target_mask.float().cuda(self.device_id)

        # Image to perform the attack on
        image_to_optimize = input_image.unsqueeze(0)
        # Copied version of image for l2 dist
        org_im_copy = copy.deepcopy(image_to_optimize.cpu()).cuda(self.device_id)
        for single_iter in range(total_iter):
            # Put in variable to get grads later on
            image_to_optimize = Variable(image_to_optimize.cuda(self.device_id), requires_grad=True)

            # Forward pass
            out = self.model(image_to_optimize)
            # Prediction
            pred_out = torch.argmax(out, dim=1).float()

            # L2 Loss
            l2_loss = calculate_l2_loss(org_im_copy, image_to_optimize)
            # Prediction loss
            pred_loss = self.calculate_pred_loss(target_mask, pred_out, out)
            # Total loss
            out_grad = torch.sum(pred_loss - l2_loss)
            # Backward pass
            out_grad.backward()

            # Add perturbation to image to optimize
            perturbed_im = image_to_optimize.data + (image_to_optimize.grad * pert_mul)
            # Do another forward pass to calculate new pert_mul
            perturbed_im_out = self.model(perturbed_im)

            # Discretize perturbed image to calculate stats
            perturbed_im_pred = torch.argmax(perturbed_im_out, dim=1).float()[0]
            perturbed_im_pred = perturbed_im_pred.detach().cpu().numpy()

            # Calculate performance of the attack
            # Similarities
            iou, pixel_acc = calculate_mask_similarity(perturbed_im_pred, target_mask_numpy)
            # Distances
            l2_dist, linf_dist = calculate_image_distance(org_im_copy, perturbed_im)
            # Update perturbation multiplier
            pert_mul = update_perturbation_multiplier(self.beta, self.tau, iou)

            # Update image to optimize and ensure boxt constraint
            image_to_optimize = perturbed_im.data.clamp_(0, 1)
            if single_iter % 20 == 0:
                if save_samples:
                    print("pert_mul: " + str(pert_mul))
                    f.writelines(str(pert_mul) + '\n')

                    save_prediction_image(pred_out.cpu().detach().numpy()[0], 'iter_' + str(single_iter),
                                          save_path + 'pred')
                    save_input_image(image_to_optimize.data.cpu().detach().numpy(), 'iter_' + str(single_iter),
                                     save_path + 'ae_image')
                    save_image_difference(image_to_optimize.data.cpu().detach().numpy(),
                                          org_im_copy.data.cpu().detach().numpy(),
                                          'iter_' + str(single_iter), save_path + 'perturbation')
                if verbose:
                    print('Iter:', single_iter, '\tIOU Overlap:', iou,
                          '\tPixel Accuracy:', pixel_acc,
                          '\n\t\tL2 Dist:', l2_dist,
                          '\tL_inf dist:', linf_dist)


if __name__ == '__main__':
    # Glaucoma dataset
    mask_dataset = MaskDatasetTest('../data/image', '../data/mask')
    # GPU parameters
    DEVICE_ID = 0

    # Load model, change it to where you download the model to
    model = load_model('../Mask_Seg/old_model/model_epoch_200.pt')
    model.eval()
    model.cpu()
    model.cuda(DEVICE_ID)

    # Attack parameters
    tau = 1e-7
    beta = 1e-6

    # Read images
    im_name1, im1, mask1 = mask_dataset[0]
    im_name2, im2, mask2 = mask_dataset[1]

    # Perform attack
    attack = AdaptiveOccludedFRAttack(DEVICE_ID, model, tau, beta)
    attack.perform_attack(im2, mask2, mask1, [0, 1])