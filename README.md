# Kuzushiji Seal Dataset (KuSeD)

## Seal Detection
* Example Train & Val (yolov8m):
```
yolo detect train model=yolov8m.pt data=./meta.yaml epochs=100 batch=8 imgsz=640 device=0 workers=4 optimizer=SGD lr0=0.01
```
* Example Test (yolov8m):
```
yolo val model=./runs/detect/train_YOLOv8m/weights/best.pt data=./meta.yaml split=test imgsz=640 batch=8 conf=0.25 iou=0.6 device=0 workers=4
```
