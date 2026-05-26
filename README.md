# Degraded Kuzushiji Documents with Seals (DKDS)

>[arXiv](https://arxiv.org/abs/2511.09117)
>[Project](https://ruiyangju.github.io/DKDS)

>Accepted by **IJDAR 2026**

<p align="center">
  <img src="img/fig_workflow.png" width="1024" title="details">
</p>

# :checkered_flag:Tracks & Challenge
## :one: Track 1: Kuzushiji Character and Seal Detection
Kuzushiji character and seal detection serves as a crucial preliminary step for subsequent Kuzushiji OCR and seal analysis.
However, this task is challenging because seals may suffer from ink fading (left two) or overlap with Kuzushiji characters or other seals (right two), which often leads to reduced detection accuracy.

<p align="left">
  <img src="img/fig_sealoverlap.png" width="640" title="details">
</p>

You can download our dataset for **Track 1: Kuzushiji Character and Seal Detection** from [here](https://1drv.ms/f/c/56c255dd1bb9ae9e/EtG5Wk7FIatCh5J0Y967n2oBNGj9DAMq_MdPyBO7gYq1FA?e=5SsERw).

## :two: Track 2: Document Binarization
Document binarization aims to improve the accuracy of downstream OCR systems. In this task, the objective is to remove seals while preserving, or even restoring, Kuzushiji characters as much as possible. 
This process becomes particularly challenging when the Kuzushiji characters overlap with seals.

<p align="left">
  <img src="img/fig_kuzushijioverlap.png" width="640" title="details">
</p>

You can download our dataset for **Track 2: Document Binarization** from [here](https://1drv.ms/f/c/56c255dd1bb9ae9e/Ekg5I9tnsJJNoVQJWfUO4aQBZ0AdgZ1wUbDBw3z_8FW5nw?e=qsezUG).

# Citation
If you find our paper useful in your research, please consider citing:
```
  @article{ju2025dkds,
    title={DKDS: A Benchmark Dataset of Degraded Kuzushiji Documents with Seals for Detection and Binarization},
    author={Ju, Rui-Yang and Yamashita, Kohei and Kameko, Hirotaka and Mori, Shinsuke},
    journal={arXiv preprint arXiv:2511.09117},
    year={2025}
  }
```

# Baseline Methods
## Environment
```
  conda create -n DKDS python=3.10
  pip install -r requirements.txt
```

## :one: Track 1: Kuzushiji Character and Seal Detection
### Baseline Performance for Testing-E set:
| Model | Param. | FLOPs | P<sub>Kuzushiji</sub> | R<sub>Kuzushiji</sub> | F<sub>Kuzushiji</sub> | P<sub>Seal</sub> | R<sub>Seal</sub> | F<sub>Seal</sub> |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| YOLOv8m | 25.84M | 78.7G | 96.0% | 89.2% | 92.5% | 95.5% | 96.4% | 95.9% |
| YOLOv9m | 20.01M | 76.5G | 96.6% | 89.6% | 93.0% | 99.8% | 96.4% | 98.1% |
| YOLOv10m | 15.31M | 58.9G | 94.1% | 89.6% | 91.8% | 99.6% | 96.4% | 98.0% |
| YOLO11m | 20.03M | 67.7G | 97.7% | 92.0% | 94.8% | 97.0% | 92.9% | 94.9% |

### Baseline Performance for Testing-D set:
| Model | Param. | FLOPs | P<sub>Kuzushiji</sub> | R<sub>Kuzushiji</sub> | F<sub>Kuzushiji</sub> | P<sub>Seal</sub> | R<sub>Seal</sub> | F<sub>Seal</sub> |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| YOLOv8m | 25.84M | 78.7G | 91.2% | 87.7% | 89.4% | 92.9% | 81.4% | 86.8% |
| YOLOv9m | 20.01M | 76.5G | 94.7% | 86.8% | 90.6% | 88.4% | 81.5% | 84.8% |
| YOLOv10m | 15.31M | 58.9G | 90.9% | 86.7% | 88.8% | 96.4% | 77.1% | 85.7% |
| YOLO11m | 20.03M | 67.7G | 95.3% | 88.7% | 91.9% | 98.4% | 84.3% | 90.8% |

### Baseline Performance for Testing-R set:
| Model | Param. | FLOPs | P<sub>Kuzushiji</sub> | R<sub>Kuzushiji</sub> | F<sub>Kuzushiji</sub> | P<sub>Seal</sub> | R<sub>Seal</sub> | F<sub>Seal</sub> |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| YOLOv8m | 25.84M | 78.7G | 84.1% | 83.3% | 83.7% | 75.8% | 85.7% | 80.4% |
| YOLOv9m | 20.01M | 76.5G | 91.6% | 82.1% | 86.6% | 75.5% | 77.3% | 76.4% |
| YOLOv10m | 15.31M | 58.9G | 86.3% | 79.5% | 82.8% | 65.2% | 77.3% | 70.7% |
| YOLO11m | 20.03M | 67.7G | 85.4% | 85.3% | 85.3% | 72.3% | 83.0% | 77.3% |

#### :round_pushpin:Train (YOLO):
We conducted training and validation of YOLO models using the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO framework.

The YOLO series of models are trained and evaluated using the following instructions:
```
  yolo detect train model=yolov8m.pt data=./meta.yaml epochs=100 batch=16 imgsz=640 device=0 workers=8 optimizer=SGD lr0=0.01 name=train_yolov8m
  yolo detect train model=yolov9m.pt data=./meta.yaml epochs=100 batch=16 imgsz=640 device=0 workers=8 optimizer=SGD lr0=0.01 name=train_yolov9m
  yolo detect train model=yolov10m.pt data=./meta.yaml epochs=100 batch=16 imgsz=640 device=0 workers=8 optimizer=SGD lr0=0.01 name=train_yolov10m
  yolo detect train model=yolo11m.pt data=./meta.yaml epochs=100 batch=16 imgsz=640 device=0 workers=8 optimizer=SGD lr0=0.01 name=train_yolo11m
```

#### :round_pushpin:Pretrained Models (YOLO):
You can download our pretrained models [here](https://1drv.ms/f/c/56c255dd1bb9ae9e/Er-w8GewvF1OhD1NK-3zqD4BhF15-o4Bc9txTcz-IetBBA?e=RkJFgK).
Please revise the `/path/to/data` in `meta.yaml`.

#### :round_pushpin:Test (YOLO):
* ``./valid`` is Testing-E set:
```
  yolo val model='./Pretrained Models for Seal Detection/yolov8m.pt' data=meta.yaml name=test_e_yolov8m
  yolo val model='./Pretrained Models for Seal Detection/yolov9m.pt' data=meta.yaml name=test_e_yolov9m
  yolo val model='./Pretrained Models for Seal Detection/yolov10m.pt' data=meta.yaml name=test_e_yolov10m
  yolo val model='./Pretrained Models for Seal Detection/yolo11m.pt' data=meta.yaml name=test_e_yolo11m
```

* ``./test`` is Testing-D set:
```
  yolo val model='./Pretrained Models for Seal Detection/yolov8m.pt' data=meta.yaml split='test' name=test_d_yolov8m
  yolo val model='./Pretrained Models for Seal Detection/yolov9m.pt' data=meta.yaml split='test' name=test_d_yolov9m
  yolo val model='./Pretrained Models for Seal Detection/yolov10m.pt' data=meta.yaml split='test' name=test_d_yolov10m
  yolo val model='./Pretrained Models for Seal Detection/yolo11m.pt' data=meta.yaml split='test' name=test_d_yolo11m
```

* ``./real`` is Testing-R set:
```
  yolo val model='./Pretrained Models for Seal Detection/yolov8m.pt' data=meta_real.yaml split='test' name=test_r_yolov8m
  yolo val model='./Pretrained Models for Seal Detection/yolov9m.pt' data=meta_real.yaml split='test' name=test_r_yolov9m
  yolo val model='./Pretrained Models for Seal Detection/yolov10m.pt' data=meta_real.yaml split='test' name=test_r_yolov10m
  yolo val model='./Pretrained Models for Seal Detection/yolo11m.pt' data=meta_real.yaml split='test' name=test_r_yolo11m
```

## :two: Track 2: Document Binarization
### Baseline Performance for Testing-E set:
| Model | FM | p-FM | PSNR | DRD | Avg-Score |
| :--: | :-: | :-: | :-: | :-: | :-: |
| Otsu | 63.01 | 63.31 | 11.76dB | 37.69 | 50.10 |
| Niblack | 39.13 | 41.14 | 8.44dB | 79.70 | 27.25 |
| Sauvola | 87.87 | 90.99 | 18.34dB | 7.01 | 72.55 |
| K-means + Otsu | 84.76 | 86.28 | 17.14dB | 9.90 | 69.57 |
| K-means + Niblack | 39.99 | 42.03 | 8.61dB | 76.67 | 28.49 |
| K-means + Sauvola | 88.59 | 91.48 | 18.65dB | 6.37 | 73.09 |
| Suh et al. (PR2022) | 93.09 | 93.09 | 20.99dB | 3.12 | 76.01 |
| Ju et al. (KBS2024) | 95.80 | 95.81 | 23.14dB | 1.76 | 78.25 |
| Improved cGAN (Ours) | 98.11 | 98.14 | 26.53dB | 0.82 | 80.49 |

### Baseline Performance for Testing-D set:
| Model | FM | p-FM | PSNR | DRD | Avg-Score |
| :--: | :-: | :-: | :-: | :-: | :-: |
| Otsu | 60.51 | 60.80 | 11.39dB | 40.89 | 47.95 |
| Niblack | 37.52 | 39.41 | 8.28dB | 82.29 | 25.73 |
| Sauvola | 84.00 | 87.04 | 17.08dB | 9.60 | 69.63 |
| K-means + Otsu | 68.10 | 69.04 | 13.37dB | 33.32 | 54.30 |
| K-means + Niblack | 35.98 | 37.04 | 8.27dB | 82.66 | 24.66 |
| K-means + Sauvola | 70.50 | 69.98 | 15.80dB | 14.69 | 60.40 |
| Suh et al. (PR2022) | 91.95 | 91.96 | 20.10dB | 3.85 | 75.04 |
| Ju et al. (KBS2024) | 94.86 | 94.90 | 22.11dB | 2.27 | 77.40 |
| Improved cGAN (Ours) | 97.08 | 97.13 | 24.58dB | 1.38 | 79.35 |

### Baseline Performance for Testing-R set:
| Model | FM | p-FM | PSNR | DRD | Avg-Score |
| :--: | :-: | :-: | :-: | :-: | :-: |
| Otsu | 67.85 | 69.36 | 13.05dB | 52.82 | 49.36 |
| Niblack | 39.67 | 42.01 | 8.31dB | 81.81 | 27.04 |
| Sauvola | 73.69 | 76.14 | 14.30dB | 32.84 | 57.82 |
| K-means + Otsu | 48.92 | 49.39 | 8.60dB | 77.09 | 32.46 |
| K-means + Niblack | 33.68 | 34.43 | 8.16dB | 83.82 | 23.11 |
| K-means + Sauvola | 53.00 | 52.80 | 12.98dB | 25.54 | 48.31 |
| Suh et al. (PR2022) | 81.00 | 81.53 | 16.26dB | 16.10 | 65.67 |
| Ju et al. (KBS2024) | 77.37 | 77.55 | 15.47dB | 30.55 | 59.96 |
| Improved cGAN (Ours) | 73.33 | 72.98 | 15.74dB | 31.44 | 57.65 |

### ① Traditional Algorithom:
#### :round_pushpin:Test:
For testing **traditional** algorithms, please follow the instruction below:
```
  python ./algorithm/traditional.py --image_test_dir ./Testset-E/image/ --mask_test_dir ./Testset-E/mask/ --save_root_dir ./Traditional_Testing_E_Result/
  python ./algorithm/traditional.py --image_test_dir ./Testset-D/image/ --mask_test_dir ./Testset-D/mask/ --save_root_dir ./Traditional_Testing_D_Result/
  python ./algorithm/traditional.py --image_test_dir ./Testset-R/image/ --mask_test_dir ./Testset-R/mask/ --save_root_dir ./Traditional_Testing_R_Result/
```
For testing **k-means + traditional** algorithms, please follow the instruction below:
```
  python ./algorithm/kmeans_traditional.py --image_test_dir ./Testset-E/image/ --mask_test_dir ./Testset-E/mask/ --save_root_dir ./KMeans_Testing_E_Result/
  python ./algorithm/kmeans_traditional.py --image_test_dir ./Testset-D/image/ --mask_test_dir ./Testset-D/mask/ --save_root_dir ./KMeans_Testing_D_Result/
  python ./algorithm/kmeans_traditional.py --image_test_dir ./Testset-R/image/ --mask_test_dir ./Testset-R/mask/ --save_root_dir ./KMeans_Testing_R_Result/
```

### ② Other SOTA methods:
#### :round_pushpin:Pretrained Models:
You can download the pretrained model of Suh *et al.* ([PR 2022](https://www.sciencedirect.com/science/article/abs/pii/S0031320322002916)) [here](https://1drv.ms/f/c/56c255dd1bb9ae9e/IgCLNE31DA6PRKFYZme5pDmTAVuuf6BrJPrnpIF15ktVSp4?e=Oxe79T).

You can download the pretrained model of Ju *et al.* ([KBS 2024](https://www.sciencedirect.com/science/article/abs/pii/S0950705124011766)) [here](https://1drv.ms/f/c/56c255dd1bb9ae9e/IgBJdVgrzi38S4Z6HzwNkJQ_Af1q51A3IbXJa51R5cc6Wn8).

#### :round_pushpin:Test (Suh et al.):
Please place ``Unet`` in the ``./weights/`` folder before running ``./SOTA/suh_test.py``.
```
  python ./SOTA/suh_test.py --lambda_bce 50 --base_model_name efficientnet-b5 --batch_size 16 --image_test_dir ./Testset-E/image/ --mask_test_dir ./Testset-E/mask/ --save_root_dir ./Suh_Testing_E_Result/
  python ./SOTA/suh_test.py --lambda_bce 50 --base_model_name efficientnet-b5 --batch_size 16 --image_test_dir ./Testset-D/image/ --mask_test_dir ./Testset-D/mask/ --save_root_dir ./Suh_Testing_D_Result/
  python ./SOTA/suh_test.py --lambda_bce 50 --base_model_name efficientnet-b5 --batch_size 16 --image_test_dir ./Testset-R/image/ --mask_test_dir ./Testset-R/mask/ --save_root_dir ./Suh_Testing_R_Result/
```

#### :round_pushpin:Test (Ju et al.):
Please place ``UnetPlusPlus`` in the ``./weights/`` folder before running ``./SOTA/ju_test.py``.
```
  python ./SOTA/ju_test.py --lambda_bce 50 --base_model_name efficientnet-b5 --batch_size 16 --image_test_dir ./Testset-E/image/ --mask_test_dir ./Testset-E/mask/ --save_root_dir ./Ju_Testing_E_Result/
  python ./SOTA/ju_test.py --lambda_bce 50 --base_model_name efficientnet-b5 --batch_size 16 --image_test_dir ./Testset-D/image/ --mask_test_dir ./Testset-D/mask/ --save_root_dir ./Ju_Testing_D_Result/
  python ./SOTA/ju_test.py --lambda_bce 50 --base_model_name efficientnet-b5 --batch_size 16 --image_test_dir ./Testset-R/image/ --mask_test_dir ./Testset-R/mask/ --save_root_dir ./Ju_Testing_R_Result/
```

### ③ Ours:
#### :round_pushpin:Train:
For training **our** model, please follow the instructions below:
```
  python ./Ours/image_to_512.py
  python ./Ours/gan_train.py
```

#### :round_pushpin:Pretrained Models:
You can download pretrained our model [here](https://1drv.ms/f/c/56c255dd1bb9ae9e/EvyWW4yx5e5OiqstmHWFXYMBe9z3Z3RwSB4bAMwcgkw_bg?e=3lGaP9).
Please place ``GAN_efficientnet-b5_50_0.00002`` in the ``./weights/`` folder before running ``gan_test.py``.

#### :round_pushpin:Test:
For testing **our** method, please follow the instruction below:
```
  python ./Ours/gan_test.py --image_test_dir ./Testset-E/image/ --mask_test_dir ./Testset-E/mask/ --save_root_dir ./Ours_Testing_E_Result/
  python ./Ours/gan_test.py --image_test_dir ./Testset-D/image/ --mask_test_dir ./Testset-D/mask/ --save_root_dir ./Ours_Testing_D_Result/
  python ./Ours/gan_test.py --image_test_dir ./Testset-R/image/ --mask_test_dir ./Testset-R/mask/ --save_root_dir ./Ours_Testing_R_Result/
```

# License
<img src="./img/CC-BY-SA.png" alt="CC BY-SA 4.0 License" width="100" style="vertical-align: middle;">  

This benchmark dataset is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

### Original Kuzushiji Dataset

The original Kuzushiji dataset used in this work is based on **『日本古典籍くずし字データセット』** (National Institute of Japanese Literature / CODH), provided by [ROIS-DS Center for Open Data in the Humanities (CODH)](https://codh.rois.ac.jp/), which is licensed under a [Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)](https://creativecommons.org/licenses/by-sa/4.0/).

The following is the citation of the original Kuzushiji dataset; please cite it when using our benchmark dataset:
```
  『日本古典籍くずし字データセット』 （国文研所蔵／CODH加工） doi:10.20676/00000340
```
