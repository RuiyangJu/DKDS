# Degraded Kuzushiji Documents with Seals (DKDS)
>[DKDS: A Benchmark Dataset of Degraded Kuzushiji Documents with Seals]()

<p align="center">
  <img src="img/fig_teaser.png" width="1024" title="details">
</p>

## Performance
| Model | Test Size | Param. | FLOPs | F1 Score | AP<sub>50</sub><sup>val</sup> | AP<sub>50-95</sub><sup>val</sup> | Speed |
| :--: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| YOLOv8 | 1024 | 43.61M | 164.9G | 0.62 | 63.58% | 40.40% | 7.7ms |
| YOLOv8+SA | 1024 | 43.64M | 165.4G | 0.63 | 64.25% | 41.64% | 8.0ms |
| YOLOv8+ECA | 1024 | 43.64M | 165.5G | 0.65 | 64.24% | 41.94% | 7.7ms |
| YOLOv8+GAM | 1024 | 49.29M | 183.5G | 0.65 | 64.26% | 41.00% | 12.7ms |
| YOLOv8+ResGAM | 1024 | 49.29M | 183.5G | 0.64 | 64.98% | 41.75% | 18.1ms |
| YOLOv8+ResCBAM | 1024 | 53.87M | 196.2G | 0.64 | 65.78% | 42.16% | 8.7ms |

## Citation
If you find our paper useful in your research, please consider citing:

## Environment
```
  pip install -r requirements.txt
```

## Seal Detection
* Example Train & Val (YOLOv8m):
```
  yolo detect train model=yolov8m.pt data=./meta.yaml epochs=100 batch=16 imgsz=640 device=0,1 workers=8 optimizer=SGD lr0=0.01 name=train_yolov8m
```
* Example Test (YOLOv8m):
```
  yolo val model=./runs/detect/train_yolov8m/weights/best.pt data=./meta.yaml split=test imgsz=640 batch=16 conf=0.25 iou=0.6 device=0,1 workers=8 name=test_yolov8m
```
