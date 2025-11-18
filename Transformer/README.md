# This repository is forked by https://github.com/dali92002/DocEnTR/.
```
  git clone https://github.com/dali92002/DocEnTR.git
```

## Environment
`requirements.txt`. in original repository can't fit on GPU 3090, please our new `requirements.txt`.
```
  conda create -n docentr python=3.8
  pip install -r requirements.txt
```

```
python train.py --data_path ./dataset/ --batch_size 4 --vit_model_size small --vit_patch_size 8 --epochs 100 --split_size 512 --validation_dataset valid
```

## Citation
If you find this useful for your research, please cite it as follows:

```bash
@inproceedings{souibgui2022docentr,
  title={DocEnTr: An end-to-end document image enhancement transformer},
  author={Souibgui, Mohamed Ali and Biswas, Sanket and  Jemni, Sana Khamekhem and Kessentini, Yousri and Forn{\'e}s, Alicia and Llad{\'o}s, Josep and Pal, Umapada},
  booktitle={2022 26th International Conference on Pattern Recognition (ICPR)},
  year={2022}
}
