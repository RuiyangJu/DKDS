# DocEnTR (Forked from [dali92002/DocEnTR](https://github.com/dali92002/DocEnTR/))
**ICPR 2022:** DocEnTr: An end-to-end document image enhancement transformer

## Clone Repository
```
  git clone https://github.com/dali92002/DocEnTR.git
```

## Environment Setup
* The original `requirements.txt` may not work efficiently on an NVIDIA 3090 GPU.
* Please use the provided `requirements.txt` in this repository.

```
  conda create -n docentr python=3.8
  pip install -r requirements.txt
```

## Dataset
* Revise `Resize_512` folder as follow:

```
./Resize_512/
│
├── train/                        
│   ├── 1.png
│   ├── 2.png
│   └── ...
│
├── train_gt/                    
│   ├── 1.png
│   ├── 2.png
│   └── ...
│
├── valid/                        
│   ├── 1.png
│   ├── 2.png
│   ├── ...
│
├── valid_gt/ 
│   ├── 1.png
│   ├── 2.png
│   └── ...
│
├── test/                     
│   ├── 1.png
│   ├── 2.png
│   └── ...
│
└── test_gt/                       
    ├── 1.png
    ├── 2.png
    └── ...
```

## Train
```
  python train.py --data_path ./Resize_512/ --batch_size 4 --vit_model_size small --vit_patch_size 8 --epochs 100 --split_size 512 --validation_dataset valid
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
