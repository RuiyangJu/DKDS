import os
import csv
import doxapy
import numpy as np
from tqdm import tqdm
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from Base.metrics import get_metric

def read_image(file):
    return np.array(Image.open(file).convert('L'))

def remove_red_kmeans(img, K=3, median_blur_ksize=5):
    h, w, c = img.shape
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=K, random_state=0).fit(pixels)
    labels = kmeans.labels_
    centers = kmeans.cluster_centers_

    red_index = np.argmax(centers[:, 0] - (centers[:, 1] + centers[:, 2]) / 2)
    mask = (labels == red_index).reshape(h, w).astype(np.uint8) * 255
    mask = cv2.medianBlur(mask, median_blur_ksize)
    removed = cv2.inpaint(img, mask, 3, cv2.INPAINT_TELEA)
    return removed

def batch_binarize_eval_kmeans(input_folder, mask_folder, wolf_folder, gatos_folder, K=3, median_blur_ksize=5, window_size=25):
    os.makedirs(wolf_folder, exist_ok=True)
    os.makedirs(gatos_folder, exist_ok=True)

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png','.jpg','.bmp','.tif'))]

    results = {
        'Wolf': {},
        'Gatos': {}
    }

    for image_name in tqdm(images, desc="Processing images"):
        input_path = os.path.join(input_folder, image_name)
        base_name = os.path.splitext(image_name)[0]

        mask_path_png = os.path.join(mask_folder, base_name + '.png')
        mask_path_bmp = os.path.join(mask_folder, base_name + '.bmp')
        if os.path.exists(mask_path_png):
            gt_mask = cv2.imread(mask_path_png, cv2.IMREAD_GRAYSCALE)
        elif os.path.exists(mask_path_bmp):
            gt_mask = cv2.imread(mask_path_bmp, cv2.IMREAD_GRAYSCALE)
        else:
            gt_mask = None

        try:
            img = cv2.imread(input_path)
            img_clean = remove_red_kmeans(img, K=K, median_blur_ksize=median_blur_ksize)
            gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

            binary_wolf = np.zeros_like(gray, dtype=np.uint8)
            wolf = doxapy.Binarization(doxapy.Binarization.Algorithms.WOLF)
            wolf.initialize(gray)
            wolf.to_binary(binary_wolf, {"window": window_size, "k": 0.5})
            wolf_output_path = os.path.join(wolf_folder, f"{base_name}_Wolf.png")
            cv2.imwrite(wolf_output_path, binary_wolf)

            binary_gatos = np.zeros_like(gray, dtype=np.uint8)
            gatos = doxapy.Binarization(doxapy.Binarization.Algorithms.GATOS)
            gatos.initialize(gray)
            gatos.to_binary(binary_gatos, {"window": window_size, "k": 0.2})
            gatos_output_path = os.path.join(gatos_folder, f"{base_name}_Gatos.png")
            cv2.imwrite(gatos_output_path, binary_gatos)

            for method_name, bin_img in zip(['Wolf','Gatos'], [binary_wolf, binary_gatos]):
                if gt_mask is not None:
                    gt_mask_bin = (gt_mask > 0).astype(np.uint8)
                    bin_mask = (bin_img > 0).astype(np.uint8)
                    F, PF, PSNR, DRD = get_metric(bin_mask, gt_mask_bin)

                    if base_name not in results[method_name]:
                        results[method_name][base_name] = {'F': [], 'PF': [], 'PSNR': [], 'DRD': []}
                    results[method_name][base_name]['F'].append(F)
                    results[method_name][base_name]['PF'].append(PF)
                    results[method_name][base_name]['PSNR'].append(PSNR)
                    results[method_name][base_name]['DRD'].append(DRD)

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    save_csv_file = os.path.join(os.path.dirname(wolf_folder), 'metrics.csv')
    with open(save_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Image', 'F-Measure', 'P-Fmeasure', 'PSNR', 'DRD'])
        for method_name in ['Wolf','Gatos']:
            datasets = sorted(results[method_name].keys())
            for img_name in datasets:
                vals = results[method_name][img_name]
                F_avg = np.mean(vals['F'])
                PF_avg = np.mean(vals['PF'])
                PSNR_avg = np.mean(vals['PSNR'])
                DRD_avg = np.mean(vals['DRD'])
                writer.writerow([method_name, img_name, F_avg, PF_avg, PSNR_avg, DRD_avg])

            all_F = np.concatenate([results[method_name][d]['F'] for d in datasets])
            all_PF = np.concatenate([results[method_name][d]['PF'] for d in datasets])
            all_PSNR = np.concatenate([results[method_name][d]['PSNR'] for d in datasets])
            all_DRD = np.concatenate([results[method_name][d]['DRD'] for d in datasets])
            writer.writerow([method_name, 'average', np.mean(all_F), np.mean(all_PF), np.mean(all_PSNR), np.mean(all_DRD)])

    print("Processing done! CSV and binary images saved in:", os.path.dirname(wolf_folder))


if __name__ == "__main__":
    input_folder = "./Testset/image"
    mask_folder = "./Testset/mask"
    wolf_folder = "./KMeans_DoxaPy_Result/Wolf"
    gatos_folder = "./KMeans_DoxaPy_Result/Gatos"

    batch_binarize_eval_kmeans(input_folder, mask_folder, wolf_folder, gatos_folder)
