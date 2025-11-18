import os
import cv2
import numpy as np
from tqdm import tqdm
import random

def prepare_dibco_experiment(input_root, output_root, train_folder, val_folder, test_folder, patch_size_train, patch_size_valid, overlap_size):
    for folder in ['train', 'train_gt', 'valid', 'valid_gt', 'test', 'test_gt']:
        os.makedirs(os.path.join(output_root, folder), exist_ok=True)

    n_i = 1 

    def process_folder(folder_name, patch_size, stride, is_train=True):
        nonlocal n_i
        folder_path = os.path.join(input_root, folder_name)
        imgs_path = os.path.join(folder_path, 'imgs')
        gts_path = os.path.join(folder_path, 'gt_imgs')

        img_files = [f for f in os.listdir(imgs_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

        for img_file in tqdm(img_files, desc=f'Processing {folder_name}'):
            img_path = os.path.join(imgs_path, img_file)
            gt_path = os.path.join(gts_path, os.path.splitext(img_file)[0] + '.png')

            img = cv2.imread(img_path)
            gt_img = cv2.imread(gt_path)

            if img is None:
                print(f"Warning: image not found {img_path}")
                continue
            if gt_img is None:
                print(f"Warning: gt image not found {gt_path}, using white image")
                gt_img = np.ones_like(img, dtype=np.uint8) * 255

            h, w = img.shape[:2]

            for i in range(0, h, stride):
                for j in range(0, w, stride):
                    if i + patch_size <= h and j + patch_size <= w:
                        p = img[i:i+patch_size, j:j+patch_size, :]
                        gt_p = gt_img[i:i+patch_size, j:j+patch_size, :]
                    else:
                        p = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 255
                        gt_p = np.ones((patch_size, patch_size, 3), dtype=np.uint8) * 255
                        hi = min(patch_size, h - i)
                        wi = min(patch_size, w - j)
                        p[0:hi, 0:wi, :] = img[i:i+hi, j:j+wi, :]
                        gt_p[0:hi, 0:wi, :] = gt_img[i:i+hi, j:j+wi, :]

                    if folder_name == train_folder:
                        cv2.imwrite(os.path.join(output_root, 'train', f'{n_i}.png'), p)
                        cv2.imwrite(os.path.join(output_root, 'train_gt', f'{n_i}.png'), gt_p)
                        n_i += 1
                    elif folder_name == val_folder:
                        out_name = f"{os.path.splitext(img_file)[0]}_{i}_{j}.png"
                        cv2.imwrite(os.path.join(output_root, 'valid', out_name), p)
                        cv2.imwrite(os.path.join(output_root, 'valid_gt', out_name), gt_p)
                    elif folder_name == test_folder:
                        out_name = f"{os.path.splitext(img_file)[0]}_{i}_{j}.png"
                        cv2.imwrite(os.path.join(output_root, 'test', out_name), p)
                        cv2.imwrite(os.path.join(output_root, 'test_gt', out_name), gt_p)

    process_folder(train_folder, patch_size_train, overlap_size, is_train=True)
    process_folder(val_folder, patch_size_valid, patch_size_valid, is_train=False)
    process_folder(test_folder, patch_size_valid, patch_size_valid, is_train=False)


if __name__ == "__main__":
    # 設定參數
    input_root = './dataset' 
    output_root = './processed' 
    train_folder = 'train'
    val_folder = 'valid'
    test_folder = 'test'

    split_size = 512
    patch_size_train = split_size + 128
    patch_size_valid = split_size
    overlap_size = split_size // 2

    prepare_dibco_experiment(input_root, output_root, train_folder, val_folder, test_folder,
                              patch_size_train, patch_size_valid, overlap_size)

    print("Processing finished!")
