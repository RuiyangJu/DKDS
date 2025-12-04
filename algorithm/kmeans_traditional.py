import os
import cv2
import numpy as np
import csv
from tqdm import tqdm
import argparse
from Base.metrics import get_metric
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu, threshold_niblack, threshold_sauvola

parser = argparse.ArgumentParser()

parser.add_argument('--image_test_dir', type=str, default='./Testset/image/', help='Directory of test images')
parser.add_argument('--mask_test_dir', type=str, default='./Testset/mask/', help='Directory of ground truth masks')
parser.add_argument('--save_root_dir', type=str, default='./KMeans_Result', help='Directory to save results')

args = parser.parse_args()

image_test_dir = args.image_test_dir
mask_test_dir = args.mask_test_dir
save_root_dir = args.save_root_dir

os.makedirs(save_root_dir, exist_ok=True)

K = 3
median_blur_ksize = 5
window_size = 25

images = [f for f in os.listdir(image_test_dir) if f.lower().endswith(('.png','.jpg','.bmp','.tif'))]

results = {
    'Otsu': {},
    'Niblack': {},
    'Sauvola': {}
}

for method in results.keys():
    os.makedirs(os.path.join(save_root_dir, method), exist_ok=True)

for image_name in tqdm(images):
    img_path = os.path.join(image_test_dir, image_name)
    img = cv2.imread(img_path)
    h, w, c = img.shape
    pixels = img.reshape(-1, 3)

    kmeans = KMeans(n_clusters=K, random_state=0).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_
    red_index = np.argmax(centers[:, 0] - (centers[:, 1] + centers[:, 2]) / 2)
    mask = (labels == red_index).reshape(h, w).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, median_blur_ksize)
    removed = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)

    gray = cv2.cvtColor(removed, cv2.COLOR_BGR2GRAY)

    methods = {
        'Otsu': (gray > threshold_otsu(gray)).astype(np.uint8) * 255,
        'Niblack': (gray > threshold_niblack(gray, window_size=window_size, k=0.8)).astype(np.uint8) * 255,
        'Sauvola': (gray > threshold_sauvola(gray, window_size=window_size)).astype(np.uint8) * 255
    }

    base_name = os.path.splitext(image_name)[0]

    mask_path_png = os.path.join(mask_test_dir, base_name + '.png')
    mask_path_bmp = os.path.join(mask_test_dir, base_name + '.bmp')
    if os.path.exists(mask_path_png):
        gt_mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE)
    elif os.path.exists(mask_path_bmp):
        gt_mask = cv2.imread(mask_path_bmp, cv2.IMREAD_GRAYSCALE)
    else:
        gt_mask = None

    for method_name, bin_img in methods.items():
        save_path = os.path.join(save_root_dir, method_name, base_name + "_bin.png")
        cv2.imwrite(save_path, bin_img)

        if gt_mask is not None:
            gt_mask_bin = gt_mask.copy()
            gt_mask_bin[gt_mask_bin > 0] = 1
            bin_mask = bin_img.copy()
            bin_mask[bin_mask > 0] = 1
            F, PF, PSNR, DRD = get_metric(bin_mask, gt_mask_bin)

            if base_name not in results[method_name]:
                results[method_name][base_name] = {'F': [], 'PF': [], 'PSNR': [], 'DRD': []}
            results[method_name][base_name]['F'].append(F)
            results[method_name][base_name]['PF'].append(PF)
            results[method_name][base_name]['PSNR'].append(PSNR)
            results[method_name][base_name]['DRD'].append(DRD)

for method_name in results.keys():
    csv_path = os.path.join(save_root_dir, f'metrics_{method_name}.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Image', 'F-Measure', 'P-Fmeasure', 'PSNR', 'DRD'])
        datasets = sorted(results[method_name].keys())
        for img_name in datasets:
            vals = results[method_name][img_name]
            F_avg = np.mean(vals['F'])
            PF_avg = np.mean(vals['PF'])
            PSNR_avg = np.mean(vals['PSNR'])
            DRD_avg = np.mean(vals['DRD'])
            writer.writerow([img_name, F_avg, PF_avg, PSNR_avg, DRD_avg])

        all_F = np.concatenate([results[method_name][d]['F'] for d in datasets])
        all_PF = np.concatenate([results[method_name][d]['PF'] for d in datasets])
        all_PSNR = np.concatenate([results[method_name][d]['PSNR'] for d in datasets])
        all_DRD = np.concatenate([results[method_name][d]['DRD'] for d in datasets])
        writer.writerow(['average', np.mean(all_F), np.mean(all_PF), np.mean(all_PSNR), np.mean(all_DRD)])


print("Processing done! CSV and k-means + binary images saved in:", save_root_dir)
