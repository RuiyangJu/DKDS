import os
import numpy as np
import math
import cv2
from config import Configs

cfg = Configs().parse() 
SPLITSIZE = cfg.split_size

def imvisualize(imdeg, imgt, impred, ind, epoch='0', setting=''):
    """
    Visualize predicted images (full) with degraded and GT images
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # Tensor -> numpy, transpose to HWC
    imdeg = np.transpose(imdeg.cpu().numpy(), (1, 2, 0))
    imgt = np.transpose(imgt.cpu().numpy(), (1, 2, 0))
    impred = np.transpose(impred.cpu().numpy(), (1, 2, 0))

    # Unnormalize
    for ch in range(3):
        imdeg[:,:,ch] = imdeg[:,:,ch] * std[ch] + mean[ch]
        imgt[:,:,ch] = imgt[:,:,ch] * std[ch] + mean[ch]
        impred[:,:,ch] = impred[:,:,ch] * std[ch] + mean[ch]

    impred = np.clip(impred, 0, 1)
    impred = (impred > 0.5).astype(np.float32)  # binarize

    out_dir = os.path.join('vis' + setting, 'epoch' + str(epoch))
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, ind + '_deg.png'), (imdeg*255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, ind + '_gt.png'), (imgt*255).astype(np.uint8))
    cv2.imwrite(os.path.join(out_dir, ind + '_pred.png'), (impred*255).astype(np.uint8))

def psnr(img1, img2):
    """
    Compute PSNR
    """
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return 100
    return 20 * math.log10(1.0 / math.sqrt(mse))

def count_psnr(model, validloader, epoch, setting='', device='cuda', thresh=0.5):
    """
    Compute PSNR on full validation dataset
    model: trained model
    validloader: DataLoader
    """
    total_psnr = 0
    count = 0

    model.eval()
    out_dir = os.path.join('vis' + setting, 'epoch' + str(epoch), 'reconstr_full')
    os.makedirs(out_dir, exist_ok=True)

    with torch.no_grad():
        for indices, inputs, targets in validloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # model prediction
            _, _, pred_pixels = model(inputs, targets)
            # reshape patches -> full images if necessary
            # here we assume pred_pixels is already [B, C, H, W]
            rec_images = pred_pixels  # [B, C, H, W]

            for i in range(len(inputs)):
                im_name = str(indices[i].item())
                # visualize and save
                imvisualize_full(inputs[i], targets[i], rec_images[i], im_name, epoch=str(epoch), setting=setting)
                # compute PSNR
                pred_np = np.transpose(rec_images[i].cpu().numpy(), (1,2,0))
                pred_np = (pred_np > thresh).astype(np.float32)
                target_np = np.transpose(targets[i].cpu().numpy(), (1,2,0))
                total_psnr += psnr(pred_np, target_np)
                count += 1

    avg_psnr = total_psnr / max(count, 1)
    return avg_psnr
