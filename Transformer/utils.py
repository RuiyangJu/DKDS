import os
import numpy as np
import math
import cv2
from config import Configs

cfg = Configs().parse() 
SPLITSIZE  = cfg.split_size

def imvisualize(imdeg, imgt, impred, ind, epoch='0', setting=''):
    """
    Visualize the predicted images along with the degraded and clean gt ones
    """
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    imdeg = np.transpose(imdeg.numpy(), (1,2,0))
    imgt = np.transpose(imgt.numpy(), (1,2,0))
    impred = np.transpose(impred.numpy(), (1,2,0))

    for ch in range(3):
        imdeg[:,:,ch] = (imdeg[:,:,ch]*std[ch]) + mean[ch]
        imgt[:,:,ch] = (imgt[:,:,ch]*std[ch]) + mean[ch]
        impred[:,:,ch] = (impred[:,:,ch]*std[ch]) + mean[ch]

    impred = np.clip(impred, 0, 1)
    impred = (impred>0.5)*1

    out_dir = os.path.join('vis'+setting, 'epoch'+epoch)
    os.makedirs(out_dir, exist_ok=True)

    cv2.imwrite(os.path.join(out_dir, ind.split('.')[0]+'_deg.png'), imdeg*255)
    cv2.imwrite(os.path.join(out_dir, ind.split('.')[0]+'_gt.png'), imgt*255)
    cv2.imwrite(os.path.join(out_dir, ind.split('.')[0]+'_pred.png'), impred*255)

def psnr(img1, img2):
    """
    Compute PSNR between two images
    """
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1.0
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def reconstruct(idx, h, w, epoch, setting, flipped=False):
    """
    Reconstruct full image from patches
    """
    rec_image = np.zeros(((h//SPLITSIZE + 1)*SPLITSIZE, (w//SPLITSIZE + 1)*SPLITSIZE, 3), dtype=np.float32)

    for i in range(0, h, SPLITSIZE):
        for j in range(0, w, SPLITSIZE):
            patch_path = os.path.join('vis'+setting, 'epoch'+str(epoch), f"{idx}_{i}_{j}_pred.png")
            if os.path.exists(patch_path):
                p = cv2.imread(patch_path)
                if flipped:
                    p = cv2.rotate(p, cv2.ROTATE_180)
                hi = min(SPLITSIZE, h - i)
                wi = min(SPLITSIZE, w - j)
                rec_image[i:i+hi, j:j+wi, :] = p[0:hi, 0:wi, :]
            else:
                hi = min(SPLITSIZE, h - i)
                wi = min(SPLITSIZE, w - j)
                rec_image[i:i+hi, j:j+wi, :] = 255

    return rec_image[:h, :w, :]

def count_psnr(epoch, data_path, valid_data='valid', setting='', flipped=False, thresh=0.5):
    """
    Reconstruct images and compute PSNR for full dataset
    """
    total_psnr = 0
    count = 0

    if valid_data == 'train':
        gt_folder = os.path.join(data_path, 'train_gt')
    elif valid_data == 'valid':
        gt_folder = os.path.join(data_path, 'valid_gt')
    elif valid_data == 'test':
        gt_folder = os.path.join(data_path, 'test_gt')
    else:
        gt_folder = os.path.join(data_path, valid_data, 'gt_imgs')

    gt_imgs = [f for f in os.listdir(gt_folder) if f.lower().endswith(('.png','.jpg','.jpeg'))]

    recon_dir = os.path.join('vis'+setting, 'epoch'+str(epoch), '00_reconstr_'+('flipped' if flipped else 'normal'))
    os.makedirs(recon_dir, exist_ok=True)

    for im_file in gt_imgs:
        gt_image_path = os.path.join(gt_folder, im_file)
        gt_image = cv2.imread(gt_image_path)
        if gt_image is None:
            print(f"Warning: GT image not found: {gt_image_path}, skipping")
            continue
        max_p = np.max(gt_image)
        if max_p == 0:
            max_p = 1
        gt_image = gt_image / max_p

        pred_image = reconstruct(os.path.splitext(im_file)[0], gt_image.shape[0], gt_image.shape[1], epoch, setting, flipped) / max_p
        pred_image = (pred_image > thresh)*1

        total_psnr += psnr(pred_image, gt_image)
        count += 1

        cv2.imwrite(os.path.join(recon_dir, im_file), (gt_image*255).astype(np.uint8))
        cv2.imwrite(os.path.join(recon_dir, os.path.splitext(im_file)[0]+'_pred.png'), (pred_image*255).astype(np.uint8))

    avg_psnr = total_psnr / max(count,1)
    return avg_psnr
