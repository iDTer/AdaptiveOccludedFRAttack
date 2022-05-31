from mask_dataset import MaskDatasetTest
from util import load_model
from AOA2 import AdaptiveOccludedFRAttack


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
