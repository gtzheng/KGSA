# Knowledge-Guided Semantics Adjustment (KGSA)
This is the code for the ICDM 2022 Paper: *Knowledge-Guided Semantics Adjustment for Improved Few-Shot Classification*.

## Requirements
- Python3
- Pytorch==1.7.1

## How to Use
Datasets:
1. [miniImageNet](https://drive.google.com/file/d/16V_ZlkW4SsnNDtnGmaBRq2OoPmUOc5mY/view)
2. [tieredImageNet](https://drive.google.com/file/d/1g1aIDy2Ar_MViF2gDXFYDBTR-HYecV07/view)

Rename the folder for the miniImageNet dataset as `miniImagenet` and rename the folder for the  tieredImageNet dataset as`tiered_imagenet`. The `root_path` in the dataset configuration is the parent folder of a dataset folder.


To pre-train the embedding network on the miniImageNet and the tieredImageNet, run
```
python train_classifier.py --config configs/train_classifier_mini.yaml
python train_classifier.py --config configs/train_classifier_tiered.yaml
```

To train the KGSA on the miniImageNet and the tieredImageNet, run
```
python train_kgsa.py --config configs/train_kgsa_mini.yaml
python train_kgsa.py --config configs/train_kgsa_tiered.yaml
```
## Citation
```
@inproceedings{zheng2022knowledge,
  title={Knowledge-Guided Semantics Adjustment for Improved Few-Shot Classification},
  author={Zheng, Guangtao and Zhang, Aidong},
  booktitle={2022 IEEE International Conference on Data Mining (ICDM)},
  pages={1347--1352},
  year={2022},
  organization={IEEE}
}

```
