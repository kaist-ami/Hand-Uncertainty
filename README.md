# Hand-Uncertainty
[BMVC'25] Official repository for "Learning Correlation-aware Aleatoric Uncertainty for 3D Hand Pose Estimation"

# 📣 Getting started

## Model checkpoints
You can download model checkpoints from [ours and baselines](https://drive.google.com/drive/folders/1EZv38cIjQ4PGl1YhdOjTEo8Z4hU2utzK?usp=sharing).
After downloading the models, place them in `./checkpoints`.

```
hand_uncertainty/
└── checkpoints/
    ├── hamer_ours.ckpt
    ├── hamer_diag.ckpt
    ├── hamer_full.ckpt
    └── hamer_ours_wo_linear.ckpt
```

## Installation
Create and activate a virtual environment to work in:
```
conda create --n hand_uncertainty
conda activate hand_uncertainty
pip install -r requirements.txt
```

## HaMeR data and model preparation
Follow the instructions in [HaMeR](https://github.com/geopavlakos/hamer) to prepare trained hamer models, MANO model, hamer training data and hamer evaluation data.

# 🚀 Training
You need to change the model type `model_type` in the [code](https://github.com/kaist-ami/Hand-Uncertainty/blob/main/hamer_uncertainty/configs_hydra/train.yaml) and index of GPU device `devices` in the [code](https://github.com/kaist-ami/Hand-Uncertainty/blob/main/hamer_uncertainty/configs_hydra/trainer/gpu.yaml).  

[Model Type]
* **ours**: our proposed correlation-aware uncertainty parameterization
* **diag**: diagonal covariance parameterization
* **full**: full covariance parameterization
* **ours_wo_linear**: removing the linear layer from our parameterization

You can pass experiment name ${EXP_NAME} as an argument to the script.
```
python train.py exp_name=${EXP_NAME} experiment=hamer_vit_transformer trainer=gpu launcher=local
```

# 🧪 Testing
## Prepare evaluation dataset
Download FreiHAND evaluation set and HO-3D evaluation set from [FreiHAND](https://github.com/lmb-freiburg/freihand) and [HO-3D](https://codalab.lisn.upsaclay.fr/competitions/4318) and place them in `uncertainty_eval/freihand/gt/` and `uncertainty_eval/ho3d/gt/`.
```
hand_uncertainty/
└── uncertainty_eval/
    ├── freihand/
    │   └─ gt/
    │       ├── evaluation_verts.json
    │       └── evaluation_xyz.json
    ├── ho3d/
    │   └─ gt/
    │       ├── evaluation_verts.json
    │       └── evaluation_xyz.json
    ├── ...        
    └── ...        
```

## Evaluation
Run evaluation on FreiHAND and HO-3D datasets as follows, results are stored in `results/`.  
You need to change the model checkpoint path `ckpt_path`, model type `model_type` and experiment name `exp_name` in the [code](https://github.com/kaist-ami/Hand-Uncertainty/blob/main/hamer_uncertainty/configs_hydra/model_config.yaml).  
  
```
python eval.py 
python eval_uncertainty.py 
```

After running the commands, the `results/` directory should look like:
```
hand_uncertainty/
└── results/
    └── ${EXP_NAME}/
        ├── freihand-val.json
        ├── freihand-val_uncertainty.json
        ├── ho3d-val.json
        └── ho3d-val_uncertainty.json
```

## Evaluate hand pose estimation performance
For [FreiHAND](https://github.com/lmb-freiburg/freihand) and [HO-3D](https://codalab.lisn.upsaclay.fr/competitions/4318), `freihand-val.json` and `ho3d-val.json` prediction files stored in `results/` can be used for evaluation using their corresponding evaluation processes.

## Evaluate uncertainty estimation performance
Run below command to evaluate AUSC, AUSE and pearson correlation.  
You need to pass experiment name ${EXP_NAME} and directory where the `.json` prediction files are stored ${PATH_TO_PRED_DIR} as an argument to the script.

```
cd uncertainty_eval
python eval_uncertainty.py --dataset freihand --exp ${EXP_NAME} --pred_file_dir ${PATH_TO_PRED_DIR}
python eval_uncertainty.py --dataset ho3d --exp ${EXP_NAME} --pred_file_dir ${PATH_TO_PRED_DIR}
```

Scores are saved in `uncertainty_eval/save/${DATASET}/${EXP_NAME}/scores.txt`.

# 📚 Citation
If you found this code useful, please consider citing our paper.

```
@article{chae2025learning,
  title={Learning Correlation-aware Aleatoric Uncertainty for 3D Hand Pose Estimation},
  author={Chae-Yeon, Lee and Hyeon-Woo, Nam and Oh, Tae-Hyun},
  journal={arXiv preprint arXiv:2509.01242},
  year={2025}
}
```

# 🙏 Acknowledgement
We heavily borrow the code from the following projects. We sincerely appreciate the authors of these projects for making their work publicly available:
- [HaMeR](https://github.com/geopavlakos/hamer)
- [FreiHAND](https://github.com/lmb-freiburg/freihand)
- [HO-3D](https://codalab.lisn.upsaclay.fr/competitions/4318)


