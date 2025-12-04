import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
import argparse
from Base.metrics import get_metric
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

parser = argparse.ArgumentParser()

parser.add_argument('--image_test_dir', type=str, default='./Testset/image/', help='Directory of test images')
parser.add_argument('--mask_test_dir', type=str, default='./Testset/mask/', help='Directory of ground truth masks')
parser.add_argument('--save_root_dir', type=str, default='./Traditional_Result/', help='Directory to save results')

args = parser.parse_args()

image_test_dir = args.image_test_dir
mask_test_dir = args.mask_test_dir
save_root_dir = args.save_root_dir

os.makedirs(save_root_dir, exist_ok=True)

window_size = 25

def otsu_binarization(img):
    thresh = threshold_otsu(img)
    return (img > thresh).astype(np.uint8) * 255

def niblack_binarization(img, window_size=window_size, k=0.8):
    thresh = threshold_niblack(img, window_size=window_size, k=k)
    return (img > thresh).astype(np.uint8) * 255

def sauvola_binarization(img, window_size=window_size):
    thresh = threshold_sauvola(img, window_size=window_size)
    return (img > thresh).astype(np.uint8) * 255

images = [f for f in os.listdir(image_test_dir) if f.lower().endswith(('.png','.jpg','.bmp','.tif'))]

results = {
    'Otsu': {},
    'Niblack': {},
    'Sauvola': {}
}

for method_name in ['Otsu','Niblack','Sauvola']:
    os.makedirs(os.path.join(save_root_dir, method_name), exist_ok=True)

for image_name in tqdm(images):
    img_path = os.path.join(image_test_dir, image_name)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    base_name = os.path.splitext(image_name)[0]
    mask_path_png = os.path.join(mask_test_dir, base_name + '.png')
    mask_path_bmp = os.path.join(mask_test_dir, base_name + '.bmp')
    if os.path.exists(mask_path_png):
        mask_path = mask_path_png
    elif os.path.exists(mask_path_bmp):
        mask_path = mask_path_bmp
    else:
        print(f"no mask for {image_name}")
        continue
    gt_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    gt_mask[gt_mask > 0] = 1

    dataset = os.path.basename(mask_path)

    methods = {
        'Otsu': otsu_binarization(img),
        'Niblack': niblack_binarization(img),
        'Sauvola': sauvola_binarization(img)
    }

    for method_name, bin_img in methods.items():
        method_dir = os.path.join(save_root_dir, method_name)
        save_path = os.path.join(method_dir, dataset)
        cv2.imwrite(save_path, bin_img)

        bin_mask = np.copy(bin_img)
        bin_mask[bin_mask > 0] = 1
        F, PF, PSNR, DRD = get_metric(bin_mask, gt_mask)

        if dataset not in results[method_name]:
            results[method_name][dataset] = {'F': [], 'PF': [], 'PSNR': [], 'DRD': []}
        results[method_name][dataset]['F'].append(F)
        results[method_name][dataset]['PF'].append(PF)
        results[method_name][dataset]['PSNR'].append(PSNR)
        results[method_name][dataset]['DRD'].append(DRD)

save_csv_file = os.path.join(save_root_dir, 'metrics.csv')
with open(save_csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Method', 'F-Measure', 'P-Fmeasure', 'PSNR', 'DRD'])

    for method_name in ['Otsu','Niblack','Sauvola']:
        datasets = sorted(results[method_name].keys())
        for dataset in datasets:
            vals = results[method_name][dataset]
            F_avg = np.mean(vals['F'])
            PF_avg = np.mean(vals['PF'])
            PSNR_avg = np.mean(vals['PSNR'])
            DRD_avg = np.mean(vals['DRD'])
            writer.writerow([dataset, F_avg, PF_avg, PSNR_avg, DRD_avg])

        all_F = np.concatenate([results[method_name][d]['F'] for d in datasets])
        all_PF = np.concatenate([results[method_name][d]['PF'] for d in datasets])
        all_PSNR = np.concatenate([results[method_name][d]['PSNR'] for d in datasets])
        all_DRD = np.concatenate([results[method_name][d]['DRD'] for d in datasets])
        writer.writerow(['average', np.mean(all_F), np.mean(all_PF), np.mean(all_PSNR), np.mean(all_DRD)])

print("Processing done! CSV and binary images saved in:", save_root_dir)
