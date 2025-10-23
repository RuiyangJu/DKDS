# Degraded Kuzushiji Documents with Seals (DKDS)
>[DKDS: A Benchmark Dataset of Degraded Kuzushiji Documents with Seals for Detection and Binarization]()

## Teaser
<p align="center">
  <img src="img/fig_teaser.png" width="1024" title="details">
</p>

## Tracks & Challenge
### Track 1: Text and Seal Detection
Seals may overlap with Kuzushiji characters (or other seals) and appear with faint ink, adversely affecting detection accuracy. 

You can download our dataset for **Track 1: Text and Seal Detection** from [here](https://1drv.ms/f/c/56c255dd1bb9ae9e/EhOn3QlrciNLtCXkdwY2y5oBbcjT6zlbpUaA7Xuj-DkFfg?e=N3dXWE).
<p align="left">
  <img src="img/fig_track1.png" width="640" title="details">
</p>

### Track 2: Document Binarization
When seals overlap with Kuzushiji characters, removing the seals while preserving (or even restoring) the underlying characters poses a significant challenge.

You can download our dataset for **Track 2: Document Binarization** from [here](https://1drv.ms/f/c/56c255dd1bb9ae9e/Ekg5I9tnsJJNoVQJWfUO4aQBZ0AdgZ1wUbDBw3z_8FW5nw?e=qsezUG).
<p align="left">
  <img src="img/fig_track2.png" width="640" title="details">
</p>

## Citation
If you find our paper useful in your research, please consider citing:
```
```

## Environment
```
  pip install -r requirements.txt
```

## Seal Detection
### Baseline Performance
| Model     | Param.   | FLOPs    | AP<sub>50</sub><sup>Kuzushiji</sup> | AP<sub>50:95</sub><sup>Kuzushiji</sup> | AP<sub>50</sub><sup>Seal</sup> | AP<sub>50:95</sub><sup>Seal</sup> |
| :--:      | :-:      | :-:      | :-:                                  | :-:                                    | :-:                             | :-:                                |
| YOLOv8m   | 25.86M   | 79.1G    | 96.4%                                | 71.2%                                  | 99.1%                           | 86.2%                              |
| YOLOv9m   | 20.16M   | 77.5G    | 96.3%                                | 71.7%                                  | 97.2%                           | 81.4%                              |
| YOLOv10m  | 16.49M   | 64.0G    | 96.2%                                | 71.4%                                  | 99.1%                           | 85.7%                              |
| YOLOv11m  | 20.05M   | 68.2G    | 97.8%                                | 74.1%                                  | 98.5%                           | 85.7%                              |

### Pretrained Models
You can download our pretrained models [here](https://1drv.ms/f/c/56c255dd1bb9ae9e/Er-w8GewvF1OhD1NK-3zqD4BhF15-o4Bc9txTcz-IetBBA?e=RkJFgK).

### Train & Evaluate
We conducted training and validation of YOLO models using the [Ultralytics](https://github.com/ultralytics/ultralytics) YOLO framework.
The YOLO series of models are trained and evaluated using the following instructions:
```
  yolo detect train model=yolov8m.pt data=./meta.yaml epochs=100 batch=16 imgsz=640 device=0 workers=8 optimizer=SGD lr0=0.01 name=train_yolov8m
  yolo detect train model=yolov9m.pt data=./meta.yaml epochs=100 batch=16 imgsz=640 device=0 workers=8 optimizer=SGD lr0=0.01 name=train_yolov9m
  yolo detect train model=yolov10m.pt data=./meta.yaml epochs=100 batch=16 imgsz=640 device=0 workers=8 optimizer=SGD lr0=0.01 name=train_yolov10m
  yolo detect train model=yolo11m.pt data=./meta.yaml epochs=100 batch=16 imgsz=640 device=0 workers=8 optimizer=SGD lr0=0.01 name=train_yolo11m
```

## Document Binarization
### Baseline Performance
| Model | FM | p-FM | PSNR | DRD | Avg-Score |
| :--: | :-: | :-: | :-: | :-: | :-: |
| Niblack | 39.13 | 41.14 | 8.44dB | 79.70 | 27.25 |
| Otsu | 63.01 | 63.31 | 11.76dB | 37.69 | 50.10 |
| Sauvola | 87.87 | 90.99 | 18.34dB | 7.01 | 72.55 |
| K-means + Niblack | 39.99 | 42.03 | 8.61dB | 76.67 | 28.49 |
| K-means + Otsu | 84.76 | 86.28 | 17.14dB | 9.90 | 69.57 |
| K-means + Sauvola | 88.59 | 91.48 | 18.65dB | 6.37 | 73.09 |
| GAN | 98.11 | 98.14 | 26.53dB | 0.82 | 80.49 |

### Pretrained Models
You can download our pretrained models [here](https://1drv.ms/f/c/56c255dd1bb9ae9e/EvyWW4yx5e5OiqstmHWFXYMBe9z3Z3RwSB4bAMwcgkw_bg?e=3lGaP9).
Please place ``GAN_efficientnet-b5_50_0.00002`` in the ``weights`` folder before running ``gan_test.py``.

### Train & Evaluate
For traditional algorithms, please follow the instruction below:
```
  python traditional.py
```
For k-means + traditional algorithms, please follow the instruction below:
```
  python kmeans.py
```
For train and test the GAN model, please follow the instructions below:
```
  python image_to_512.py
  python gan_train.py
  python gan_test.py
```
