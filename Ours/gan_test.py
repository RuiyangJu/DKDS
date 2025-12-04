import os
import cv2
import csv
import torch
import time
import argparse
import numpy as np
from tqdm import tqdm
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from Base.tool_clean import check_is_image, get_image_patch, image_padding
from Base.gan_metrics import get_metric

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default='0', help="GPU number")
parser.add_argument('--base_model_name', type=str, default='efficientnet-b5', help='base_model_name')
parser.add_argument('--encoder_weights', type=str, default='imagenet', help='encoder_weights')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--threshold', type=float, default=0.5, help='binarization threshold, 0~1')
parser.add_argument('--image_test_dir', type=str, default='./Testset/image/', help='test image dir')
parser.add_argument('--mask_test_dir', type=str, default='./Testset/mask/', help='test mask dir')
parser.add_argument('--save_root_dir', type=str, default='./GAN_Predicted_Images', help='folder to save predicted masks and metrics')
parser.add_argument('--weight_folder', type=str, default='./weights/GAN_efficientnet-b5_50_0.0002/', help='weight folder')
parser.add_argument('--resize_global', action='store_true', help='whether to use full-image resize inference')
parser.add_argument('--resize_size', type=int, default=512, help='size for global resize')
opt = parser.parse_args()

device = torch.device(f"cuda:{opt.gpu}")

weight_list = sorted([os.path.join(opt.weight_folder, f) for f in os.listdir(opt.weight_folder) if 'unet_patch' in f])
print('stage3 patch weight:', weight_list)

model = smp.Unet(opt.base_model_name, encoder_weights=opt.encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[0], map_location='cpu'))
model.to(device)
model.eval()

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params/1e6:.2f} M")

preprocess_input = get_preprocessing_fn(opt.base_model_name, pretrained=opt.encoder_weights)

save_root_dir = opt.save_root_dir
os.makedirs(save_root_dir, exist_ok=True)

save_csv = open(os.path.join(save_root_dir, 'metrics.csv'), 'w', newline='')
save_csv_file = csv.writer(save_csv)
save_csv_file.writerow(['Image', 'F-Measure', 'P-FMeasure', 'PSNR', 'DRD'])

images = os.listdir(opt.image_test_dir)
test_images = []
for img in tqdm(images, desc="Checking images"):
    if not check_is_image(img):
        continue
    img_name = img.split('.')[0]
    gt_path_png = os.path.join(opt.mask_test_dir, img_name + '.png')
    gt_path_bmp = os.path.join(opt.mask_test_dir, img_name + '.bmp')
    if os.path.isfile(gt_path_png):
        gt_mask = gt_path_png
    elif os.path.isfile(gt_path_bmp):
        gt_mask = gt_path_bmp
    else:
        print(f'{img} no mask')
        continue
    test_images.append((os.path.join(opt.image_test_dir, img), gt_mask))

threshold_value = int(256 * opt.threshold)
start_time = time.time()

all_F, all_PF, all_PSNR, all_DRD = [], [], [], []

for test_image, test_mask in tqdm(test_images, desc="Processing images"):
    img_name = os.path.basename(test_image).split('.')[0]
    image = cv2.imread(test_image)
    h, w, _ = image.shape

    gt_mask = cv2.imread(test_mask, cv2.IMREAD_GRAYSCALE)
    gt_mask[gt_mask > 0] = 1

    image_patches, poslist = get_image_patch(image, 256, 256, overlap=0.1, is_mask=False)
    color_patches = [preprocess_input(patch.astype(np.float32), input_space="BGR") for patch in image_patches]

    preds = []
    with torch.no_grad():
        for i in range(0, len(color_patches), opt.batch_size):
            batch = color_patches[i:i+opt.batch_size]
            batch_tensor = torch.from_numpy(np.array(batch)).permute(0,3,1,2).float().to(device)
            preds.extend(torch.sigmoid(model(batch_tensor)).cpu())

    out_img = np.ones((h, w, 1), dtype=np.uint8) * 255
    for i, patch_pred in enumerate(preds):
        patch_np = (patch_pred.permute(1,2,0).numpy() * 255).astype(np.uint8)
        start_h, start_w, end_h, end_w, h_shift, w_shift = poslist[i]
        h_cut = end_h - start_h
        w_cut = end_w - start_w
        out_img[start_h:end_h, start_w:end_w] = np.minimum(
            out_img[start_h:end_h, start_w:end_w],
            patch_np[h_shift:h_shift+h_cut, w_shift:w_shift+w_cut]
        )

    final_mask = np.where(out_img > threshold_value, 255, 0).astype(np.uint8)
    final_mask_bin = np.where(final_mask > 0, 1, 0)

    fmeasure, pfmeasure, psnr, drd = get_metric(final_mask_bin, gt_mask)

    save_csv_file.writerow([img_name, fmeasure, pfmeasure, psnr, drd])

    all_F.append(fmeasure)
    all_PF.append(pfmeasure)
    all_PSNR.append(psnr)
    all_DRD.append(drd)

    cv2.imwrite(os.path.join(save_root_dir, f'{img_name}.png'), final_mask)

save_csv_file.writerow([
    'average', 
    np.mean(all_F), 
    np.mean(all_PF), 
    np.mean(all_PSNR), 
    np.mean(all_DRD)
])

save_csv.close()
print(f'All done. Total time: {time.time() - start_time:.2f}s')
