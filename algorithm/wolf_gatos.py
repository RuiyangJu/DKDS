import os
import csv
import doxapy
import numpy as np
from tqdm import tqdm
from PIL import Image
from Base.metrics import get_metric

def read_image(file):
    return np.array(Image.open(file).convert('L'))

def batch_binarize_eval(input_folder, mask_folder, wolf_folder, gatos_folder):
    os.makedirs(wolf_folder, exist_ok=True)
    os.makedirs(gatos_folder, exist_ok=True)

    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png','.jpg','.bmp','.tif'))]

    results = {
        'Wolf': {},
        'Gatos': {}
    }

    for image_name in tqdm(images):
        input_path = os.path.join(input_folder, image_name)
        base_name = os.path.splitext(image_name)[0]

        mask_path_png = os.path.join(mask_folder, base_name + '.png')
        mask_path_bmp = os.path.join(mask_folder, base_name + '.bmp')
        if os.path.exists(mask_path_png):
            mask_path = mask_path_png
        elif os.path.exists(mask_path_bmp):
            mask_path = mask_path_bmp
        else:
            print(f"No mask for {image_name}, skipped.")
            continue

        try:
            grayscale_image = read_image(input_path)
            gt_mask = read_image(mask_path)
            gt_mask[gt_mask > 0] = 1

            binary_wolf = np.empty(grayscale_image.shape, grayscale_image.dtype)
            wolf = doxapy.Binarization(doxapy.Binarization.Algorithms.WOLF)
            wolf.initialize(grayscale_image)
            wolf.to_binary(binary_wolf, {"window": 25, "k": 0.5})
            wolf_output_path = os.path.join(wolf_folder, f"{base_name}_Wolf.png")
            Image.fromarray(binary_wolf).save(wolf_output_path)

            binary_gatos = np.empty(grayscale_image.shape, grayscale_image.dtype)
            gatos = doxapy.Binarization(doxapy.Binarization.Algorithms.GATOS)
            gatos.initialize(grayscale_image)
            gatos.to_binary(binary_gatos, {"window": 25, "k": 0.2})
            gatos_output_path = os.path.join(gatos_folder, f"{base_name}_Gatos.png")
            Image.fromarray(binary_gatos).save(gatos_output_path)

            for method_name, bin_img in zip(['Wolf', 'Gatos'], [binary_wolf, binary_gatos]):
                bin_mask = np.copy(bin_img)
                bin_mask[bin_mask > 0] = 1
                F, PF, PSNR, DRD = get_metric(bin_mask, gt_mask)

                if base_name not in results[method_name]:
                    results[method_name][base_name] = {'F': [], 'PF': [], 'PSNR': [], 'DRD': []}
                results[method_name][base_name]['F'].append(F)
                results[method_name][base_name]['PF'].append(PF)
                results[method_name][base_name]['PSNR'].append(PSNR)
                results[method_name][base_name]['DRD'].append(DRD)

        except Exception as e:
            print(f"Error processing {image_name}: {e}")

    save_csv_file = os.path.join(os.path.commonpath([wolf_folder, gatos_folder]), 'metrics.csv')
    with open(save_csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Method', 'Image', 'F-Measure', 'P-Fmeasure', 'PSNR', 'DRD'])

        for method_name in ['Wolf', 'Gatos']:
            datasets = sorted(results[method_name].keys())
            for dataset in datasets:
                vals = results[method_name][dataset]
                F_avg = np.mean(vals['F'])
                PF_avg = np.mean(vals['PF'])
                PSNR_avg = np.mean(vals['PSNR'])
                DRD_avg = np.mean(vals['DRD'])
                writer.writerow([method_name, dataset, F_avg, PF_avg, PSNR_avg, DRD_avg])

            all_F = np.concatenate([results[method_name][d]['F'] for d in datasets])
            all_PF = np.concatenate([results[method_name][d]['PF'] for d in datasets])
            all_PSNR = np.concatenate([results[method_name][d]['PSNR'] for d in datasets])
            all_DRD = np.concatenate([results[method_name][d]['DRD'] for d in datasets])
            writer.writerow([method_name, 'average', np.mean(all_F), np.mean(all_PF), np.mean(all_PSNR), np.mean(all_DRD)])

    print("Processing done! CSV and binary images saved.")


if __name__ == "__main__":
    input_folder = "./Testset/image"
    mask_folder = "./Testset/mask"
    wolf_folder = "./DoxaPy_Result/Wolf"
    gatos_folder = "./DoxaPy_Result/Gatos"

    batch_binarize_eval(input_folder, mask_folder, wolf_folder, gatos_folder)
