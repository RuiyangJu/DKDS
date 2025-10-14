# Kuzushiji Seal Dataset (KuSeD)

## Seal Detection
* Example Train & Val (YOLOv8m):
```
yolo detect train model=yolov8m.pt data=./meta.yaml epochs=100 batch=16 imgsz=640 device=0,1 workers=8 optimizer=SGD lr0=0.01 accumulate=4
```
* Example Test (YOLOv8m):
```
yolo val model=./runs/detect/train_YOLOv8m/weights/best.pt data=./meta.yaml split=test imgsz=640 batch=8 conf=0.25 iou=0.6 device=0 workers=4
```
