# This repository is forked by https://github.com/dali92002/DocEnTR/.
```
  git clone https://github.com/dali92002/DocEnTR.git
```

## Environment
* `requirements.txt`. in original repository can't fit on GPU 3090, please our new `requirements.txt`.

```
  conda create -n docentr python=3.8
  pip install -r requirements.txt
```

## Dataset
* Revise `Resize_512` folder as follow:

```
/your/path/
│
├── DIBCOSETS/                      ← *你原始的 DIBCO 数据集*
│   ├── 2011/                       ← (示例：每个子数据集)
│   │   ├── imgs/
│   │   └── gt_imgs/
│   ├── 2012/
│   └── ...
│
├── train/                          ← 训练图像 patch
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── ...
│
├── train_gt/                       ← 训练集 GT patch（与 train 一一对应）
│   ├── 1.png
│   ├── 2.png
│   ├── 3.png
│   └── ...
│
├── valid/                          ← 验证集图像 patch（无 overlap）
│   ├── VAL1_0_0.png
│   ├── VAL1_0_256.png
│   ├── ...
│
├── valid_gt/                       ← 验证 GT patch
│   ├── VAL1_0_0.png
│   ├── VAL1_0_256.png
│   └── ...
│
├── test/                           ← 测试集图像 patch（和 valid 同样逻辑）
│   ├── TEST1_0_0.png
│   ├── TEST1_0_256.png
│   └── ...
│
└── test_gt/                        ← 测试 GT patch
    ├── TEST1_0_0.png
    ├── TEST1_0_256.png
    └── ...
```

```
  python train.py --data_path ./dataset/ --batch_size 4 --vit_model_size small --vit_patch_size 8 --epochs 100 --split_size 512 --validation_dataset valid
```

## Citation
If you find this useful for your research, please cite it as follows:

```
    @inproceedings{souibgui2022docentr,
      title={DocEnTr: An end-to-end document image enhancement transformer},
      author={Souibgui, Mohamed Ali and Biswas, Sanket and  Jemni, Sana Khamekhem and Kessentini, Yousri and Forn{\'e}s, Alicia and Llad{\'o}s, Josep and Pal, Umapada},
      booktitle={2022 26th International Conference on Pattern Recognition (ICPR)},
      year={2022}
    }
```
