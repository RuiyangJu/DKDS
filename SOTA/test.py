import os
import numpy as np
from tqdm import tqdm
import torch
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.encoders import get_preprocessing_fn
import time
import cv2
import csv
import argparse
from Base.tool_clean import check_is_image, get_image_patch, image_padding
from Base.metrics import get_metric

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=str, default='0', help="GPU number")
parser.add_argument('--lambda_bce', type=float, default=50.0, help='bce weight')
parser.add_argument('--base_model_name', type=str, default='efficientnet-b5', help='base_model_name')
parser.add_argument('--encoder_weights', type=str, default='imagenet', help='encoder_weights')
parser.add_argument('--generator_lr', type=float, default=2e-4, help='generator learning rate')
parser.add_argument('--discriminator_lr', type=float, default=2e-4, help='discriminator learning rate')
parser.add_argument('--batch_size', type=int, default=16, help='batch size')
parser.add_argument('--threshold', type=float, default=0.3, help='binarization threshold, 0~1')

# data set
parser.add_argument('--image_test_dir', type=str, default='./Testset/image/', help='original image test dir')
parser.add_argument('--mask_test_dir', type=str, default='./Testset/mask/', help='original mask test dir')
parser.add_argument('--save_root_dir', type=str, default='./Result', help='folder to save predicted masks and metrics')

opt = parser.parse_args()

device = torch.device(f"cuda:{opt.gpu}")

base_model_name = opt.base_model_name
lambda_bce = opt.lambda_bce
generator_lr = opt.generator_lr
encoder_weights = opt.encoder_weights
threshold = opt.threshold 
batch_size = opt.batch_size

# stage 2
stage2_weight_folder = (
    './Unet/stage2_dibco_'
    + base_model_name + '_'
    + str(int(lambda_bce)) + '_'
    + str(generator_lr) + '_'
    + str(threshold) + '/'
)
weight_list = sorted(os.listdir(stage2_weight_folder))
weight_list = [
    os.path.join(stage2_weight_folder, weight_path)
    if weight_path.endswith('pth') and 'Unet' in weight_path
    else None
    for weight_path in weight_list
]
weight_list = [w for w in weight_list if w is not None]
print("Stage2 weight list:", weight_list)

models = []

# blue
model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[0], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

# green
model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[1], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

# red
model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[2], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

# gray
model = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model.load_state_dict(torch.load(weight_list[3], map_location='cpu'))
model.to(device)
model.requires_grad_(False)
model.eval()
models.append(model)

# stage 3
stage3_patch_weight_folder = (
    './Unet/stage3_dibco_'
    + base_model_name + '_'
    + str(int(lambda_bce)) + '_'
    + str(generator_lr) + '/'
)
weight_list = os.listdir(stage3_patch_weight_folder)
weight_list = [
    os.path.join(stage3_patch_weight_folder, w)
    for w in weight_list if 'unet_patch' in w
]
weight_list = sorted(weight_list)
print('Stage3 patch weight:', weight_list)

model_stage3_patch = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model_stage3_patch.load_state_dict(torch.load(weight_list[0], map_location='cpu'))
model_stage3_patch.to(device)
model_stage3_patch.requires_grad_(False)
model_stage3_patch.eval()

# stage 3 global model
stage3_resize_weight_folder = (
    './Unet/stage3_resize_dibco_'
    + base_model_name + '_'
    + str(int(lambda_bce)) + '_'
    + str(generator_lr) + '/'
)
weight_list = os.listdir(stage3_resize_weight_folder)
weight_list = [
    os.path.join(stage3_resize_weight_folder, w)
    for w in weight_list if 'unet_global' in w
]
weight_list = sorted(weight_list)
print('Stage3 resize (global) weight:', weight_list)

model_stage3_global = smp.Unet(base_model_name, encoder_weights=encoder_weights, in_channels=3)
model_stage3_global.load_state_dict(torch.load(weight_list[0], map_location='cpu'))
model_stage3_global.to(device)
model_stage3_global.requires_grad_(False)
model_stage3_global.eval()

all_models = models + [model_stage3_patch, model_stage3_global]

total_params = sum(p.numel() for m in all_models for p in m.parameters())
trainable_params = sum(p.numel() for m in all_models for p in m.parameters() if p.requires_grad)

print(f"Total parameters: {total_params/1e6:.2f} M")

image_test_dir = opt.image_test_dir
mask_test_dir = opt.mask_test_dir
preprocess_input = get_preprocessing_fn(base_model_name, pretrained=encoder_weights)

threshold_value = int(256 * threshold)
kernel = np.ones((7, 7), np.uint8)
resize_size = (512, 512)
skip_resize_ratio = 6
skip_max_length = 512
padding_resize_ratio = 4

# make directoies
save_root_dir = opt.save_root_dir
os.makedirs(save_root_dir, exist_ok=True)

# CSV file
csv_path = os.path.join(save_root_dir, 'metrics.csv')
save_csv = open(csv_path, 'w', newline='')
save_csv_file = csv.writer(save_csv)
save_csv_file.writerow(['Image', 'F-Measure', 'P-FMeasure', 'PSNR', 'DRD'])

# save fmeasure
all_F, all_PF, all_PSNR, all_DRD = [], [], [], []

images = os.listdir(image_test_dir)
test_images = []

for image in tqdm(images, desc="Scanning test images"):
    if not check_is_image(image):
        print('not image:', image)
        continue

    img_name = image.split('.')[0]
    gt_path_png = os.path.join(mask_test_dir, img_name + '.png')
    gt_path_bmp = os.path.join(mask_test_dir, img_name + '.bmp')
    if os.path.isfile(gt_path_png):
        gt_mask = gt_path_png
    elif os.path.isfile(gt_path_bmp):
        gt_mask = gt_path_bmp
    else:
        print(image, 'no mask')
        continue

    test_images.append((os.path.join(image_test_dir, image), gt_mask))

test_images = sorted(test_images, key=lambda x: os.path.basename(x[0]))

start_time = time.time()

for test_image, test_mask in tqdm(test_images, desc="Processing images"):

    img_name = os.path.basename(test_image).split('.')[0]
    image = cv2.imread(test_image)
    h, w, _ = image.shape

    gt_mask = cv2.imread(test_mask, cv2.IMREAD_GRAYSCALE)
    gt_mask[gt_mask > 0] = 1

    print('processing image:', img_name)

    image_patches, poslist = get_image_patch(image, 256, 256, overlap=0.1, is_mask=False)
    merge_img = np.ones((h, w, 3), dtype=np.uint8)
    out_imgs = []

    for channel in range(4):
        color_patches = []
        for patch in image_patches:
            tmp = patch.astype(np.float32)
            if channel != 3:
                color_patches.append(preprocess_input(tmp[:, :, channel:channel+1]))
            else:
                gray = cv2.cvtColor(tmp, cv2.COLOR_BGR2GRAY)
                gray = np.expand_dims(gray, axis=-1)
                color_patches.append(preprocess_input(gray))

        preds = []
        with torch.no_grad():
            step = 0
            while step < len(color_patches):
                ps = step
                pe = min(step + batch_size, len(color_patches))

                batch_np = np.array(color_patches[ps:pe])  
                batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(device)
                batch_pred = torch.sigmoid(models[channel](batch_tensor)).cpu()
                preds.extend(batch_pred)
                step += batch_size

        out_img = np.ones((h, w, 1), dtype=np.float32) * 255.

        for i in range(len(image_patches)):
            patch = preds[i].permute(1, 2, 0).numpy() * 255.

            start_h, start_w, end_h, end_w, h_shift, w_shift = poslist[i]
            h_cut = end_h - start_h
            w_cut = end_w - start_w

            tmp = np.minimum(
                out_img[start_h:end_h, start_w:end_w],
                patch[h_shift:h_shift + h_cut, w_shift:w_shift + w_cut]
            )
            out_img[start_h:end_h, start_w:end_w] = tmp

        out_imgs.append(out_img)

    merge_img[:, :, 0:1] = (out_imgs[0] + out_imgs[3]) / 2.
    merge_img[:, :, 1:2] = (out_imgs[1] + out_imgs[3]) / 2.
    merge_img[:, :, 2:3] = (out_imgs[2] + out_imgs[3]) / 2.
    merge_img = merge_img.astype(np.uint8)

    image_patches, poslist = get_image_patch(merge_img, 256, 256, overlap=0.1, is_mask=False)

    color_patches = []
    for patch in image_patches:
        tmp = patch.astype(np.float32)
        color_patches.append(preprocess_input(tmp, input_space="BGR"))

    preds = []
    with torch.no_grad():
        step = 0
        while step < len(color_patches):
            ps = step
            pe = min(step + batch_size, len(color_patches))

            batch_np = np.array(color_patches[ps:pe])  # NHWC
            batch_tensor = torch.from_numpy(batch_np).permute(0, 3, 1, 2).float().to(device)
            batch_pred = torch.sigmoid(model_stage3_patch(batch_tensor)).cpu()
            preds.extend(batch_pred)
            step += batch_size

    out_img = np.ones((h, w, 1), dtype=np.float32) * 255.

    for i in range(len(image_patches)):
        patch = preds[i].permute(1, 2, 0).numpy() * 255.

        start_h, start_w, end_h, end_w, h_shift, w_shift = poslist[i]
        h_cut = end_h - start_h
        w_cut = end_w - start_w

        tmp = np.minimum(
            out_img[start_h:end_h, start_w:end_w],
            patch[h_shift:h_shift + h_cut, w_shift:w_shift + w_cut]
        )
        out_img[start_h:end_h, start_w:end_w] = tmp

    stage3_out_img = out_img.astype(np.uint8)
    stage3_out_img[stage3_out_img > threshold_value] = 255
    stage3_out_img[stage3_out_img <= threshold_value] = 0
    stage3_out_img = np.squeeze(stage3_out_img, axis=-1)

    min_length = min(h, w)
    max_length = max(h, w)

    use_global = not (min_length * skip_resize_ratio < max_length or max_length < skip_max_length)

    if use_global:
        is_padded = False
        padded_size = None
        position = None

        if min_length * padding_resize_ratio < max_length:
            image_pad, position = image_padding(image)
            padded_size = image_pad.shape[0]
            is_padded = True
        else:
            image_pad = image

        resized_img = cv2.resize(image_pad, dsize=resize_size, interpolation=cv2.INTER_NEAREST)
        resized_img = preprocess_input(resized_img.astype(np.float32), input_space="BGR")
        resized_img = np.expand_dims(resized_img, axis=0)
        resized_tensor = torch.from_numpy(resized_img).permute(0, 3, 1, 2).float().to(device)

        with torch.no_grad():
            resized_mask_pred = torch.sigmoid(model_stage3_global(resized_tensor)).cpu()

        resized_mask_pred = resized_mask_pred[0].permute(1, 2, 0).numpy() * 255
        resized_mask_pred = resized_mask_pred.astype(np.uint8)
        resized_mask_pred[resized_mask_pred > threshold_value] = 255
        resized_mask_pred[resized_mask_pred <= threshold_value] = 0

        if is_padded:
            resized_mask_pred = cv2.resize(
                resized_mask_pred,
                dsize=(padded_size, padded_size),
                interpolation=cv2.INTER_NEAREST
            )
            resized_mask_pred = resized_mask_pred[
                position[0]:position[1],
                position[2]:position[3]
            ]
        else:
            resized_mask_pred = cv2.resize(
                resized_mask_pred,
                dsize=(w, h),
                interpolation=cv2.INTER_NEAREST
            )

        resized_mask_pred = cv2.erode(resized_mask_pred, kernel, iterations=1)

    if not use_global:
        final_mask = stage3_out_img
    else:
        stage3_normal_or_img = np.bitwise_or(resized_mask_pred, stage3_out_img)
        final_mask = stage3_normal_or_img

    save_path = os.path.join(save_root_dir, f'{img_name}.png')
    cv2.imwrite(save_path, final_mask)

    final_mask_bin = np.copy(final_mask)
    final_mask_bin[final_mask_bin > 0] = 1

    fmeasure, pfmeasure, psnr, drd = get_metric(final_mask_bin, gt_mask)

    save_csv_file.writerow([img_name, fmeasure, pfmeasure, psnr, drd])

    all_F.append(fmeasure)
    all_PF.append(pfmeasure)
    all_PSNR.append(psnr)
    all_DRD.append(drd)

if len(all_F) > 0:
    save_csv_file.writerow([
        'average',
        float(np.mean(all_F)),
        float(np.mean(all_PF)),
        float(np.mean(all_PSNR)),
        float(np.mean(all_DRD))
    ])

save_csv.close()
print(f'All done. Total time: {time.time() - start_time:.2f}s')
