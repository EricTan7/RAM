# RAM🚀: Open-Vocabulary Multi-Label Recognition through Knowledge-Constrained Optimal Transport

Official implementation of the paper in CVPR 2025:

**Recover and Match: Open-Vocabulary Multi-Label Recognition through Knowledge-Constrained Optimal Transport**

## 📨 Introduction

RAM is an efficient matching framework for OVMLR (Open-Vocabulary Multi-Label Recognition). To address the urgent problems in existing methods, RAM involves (1) LLA to recover regional semantics, and (2) KCOT to find precise region-to-label matching.

<p align="center">
    <img src="src/method.pdf" alt="RAM Framework" width="75%">
</p>


## 🔧 Installation

Install the environment through conda and pip is recommended:

```shell
conda create -n ram python=3.10
conda activate ram

# Install the dependencies
pip install -r requirements.txt
```


## 🎯 Running the code
- `model/model.py`: Implementation of RAM model
- `model/ot_solver.py`: Implementation of Sinkhorn Algorithm
- `clip/adapters.py`: Implementation of LLA (Local Adapter)
- `loss/mmc_loss.py`: Implementation of MMC loss (Multi-Matching loss) 

Run the following code to start training:
```shell
python train.py --config_file configs/coco.yml
```
Use wandb to log the running:
```shell
python train.py --config_file configs/coco.yml WANDB True
```

## 💬 Discussion
The core contribution is the OT-based matching pipeline, which we found beneficial to the OVMLR task while remaining highly efficient.
The matching framework can be easily extended to dense prediction tasks (e.g., **semantic segmentation**). Welcome to transfer our approach to the segmentation scenarios. 

