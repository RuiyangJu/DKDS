# Degraded Kuzushiji Documents with Seals (DKDS)
>[arXiv](https://arxiv.org/abs/2511.09117)
>[Project](https://ruiyangju.github.io/DKDS)

<p align="center">
  <img src="img/fig_teaser.png" width="1024" title="details">
</p>

# :checkered_flag:Tracks & Challenge
## :one: Track 1: Text and Seal Detection
Text and seal detection serves as a crucial preliminary step for subsequent Kuzushiji OCR and seal analysis.
However, this task is challenging because seals may suffer from ink fading (left two) or overlap with Kuzushiji characters or other seals (right two), which often leads to reduced detection accuracy.

<p align="left">
  <img src="img/fig_challenge1.png" width="640" title="details">
</p>

You can download our dataset for **Track 1: Text and Seal Detection** from [here](https://1drv.ms/f/c/56c255dd1bb9ae9e/EtG5Wk7FIatCh5J0Y967n2oBNGj9DAMq_MdPyBO7gYq1FA?e=5SsERw).

## :two: Track 2: Document Binarization
Document binarization aims to improve the accuracy of downstream OCR systems. In this task, the objective is to remove seals while preserving, or even restoring, Kuzushiji characters as much as possible. 
This process becomes particularly challenging when the Kuzushiji characters overlap with seals.

<p align="left">
  <img src="img/fig_challenge2.png" width="640" title="details">
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

## :one: Track 1: Text and Seal Detection
### Baseline Performance for Testing-D set:
| Model | Param. | FLOPs | P<sub>Kuzushiji</sub> | R<sub>Kuzushiji</sub> | F<sub>Kuzushiji</sub> | P<sub>Seal</sub> | R<sub>Seal</sub> | F<sub>Seal</sub> |
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| YOLOv8m | 25.86M | 79.1G | 90.5% | 87.9% | 89.2% | 92.9% | 81.4% | 86.8% |
| YOLOv9m | 20.16M | 77.5G | 94.0% | 87.1% | 90.4% | 88.4% | 81.4% | 84.8% |
| YOLOv10m | 16.49M | 64.0G | 90.4% | 87.1% | 88.7% | 96.4% | 77.1% | 85.7% |
| YOLO11m | 20.05M | 68.2G | 94.7% | 89.1% | 91.8% | 98.5% | 84.3% | 90.8% |

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

## :two: Track 2: Document Binarization
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

### ① Traditional Algorithom:
#### :round_pushpin:Test:
For testing **traditional** algorithms, please follow the instruction below:
```
  python ./algorithm/traditional.py --image_test_dir ./Testset-E/image/ --mask_test_dir ./Testset-E/mask/ --save_root_dir ./Traditional_Testing_E_Result/
  python ./algorithm/traditional.py --image_test_dir ./Testset-D/image/ --mask_test_dir ./Testset-D/mask/ --save_root_dir ./Traditional_Testing_D_Result/
```
For testing **k-means + traditional** algorithms, please follow the instruction below:
```
  python ./algorithm/kmeans_traditional.py --image_test_dir ./Testset-E/image/ --mask_test_dir ./Testset-E/mask/ --save_root_dir ./KMeans_Testing_E_Result/
  python ./algorithm/kmeans_traditional.py --image_test_dir ./Testset-D/image/ --mask_test_dir ./Testset-D/mask/ --save_root_dir ./KMeans_Testing_D_Result/
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
```

#### :round_pushpin:Test (Ju et al.):
Please place ``UnetPlusPlus`` in the ``./weights/`` folder before running ``./SOTA/ju_test.py``.
```
  python ./SOTA/ju_test.py --lambda_bce 50 --base_model_name efficientnet-b5 --batch_size 16 --image_test_dir ./Testset-E/image/ --mask_test_dir ./Testset-E/mask/ --save_root_dir ./Ju_Testing_E_Result/
  python ./SOTA/ju_test.py --lambda_bce 50 --base_model_name efficientnet-b5 --batch_size 16 --image_test_dir ./Testset-D/image/ --mask_test_dir ./Testset-D/mask/ --save_root_dir ./Ju_Testing_D_Result/
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
