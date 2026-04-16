# 🚗 FaithDaIL 
**Faithful Dynamic Imitation Learning from Human Intervention with Dynamic Regret Minimization**

[![Conference](https://img.shields.io/badge/NeurIPS-2025-blue.svg)](https://neurips.cc/) 
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Supported-EE4C2C.svg)](https://pytorch.org/)

Welcome to the official repository for the **NeurIPS 2025** paper **"Faithful Dynamic Imitation Learning from Human Intervention with Dynamic Regret Minimization"**.

This repository contains the core implementations of our proposed **FaithDaIL** algorithm and training pipelines for highly complex autonomous driving simulators (**CARLA** & **MetaDrive**).

---

## 🏗️ Repository Structure

```text
FaithDaIL/
├── algos/                  # Algorithm implementations
├── configs/                # Hyperparameter configurations for Carla and MetaDrive
├── envs/                   # Environment wrappers (MetaDrive & CARLA + DI-drive core)
├── networks/               # Neural network architectures (Policy, Value, Discriminators)
├── train_online.py         # Main entry point for online training
├── collect_human_data.py   # Script for collecting human intervention data
├── eval_carla_models.py    # Evaluation script for CARLA environments
├── eval_metadrive_models.py# Evaluation script for MetaDrive environments
└── requirement.txt         # Project dependencies
```

## ⚙️ Installation

To set up the environment, we recommend using a Conda virtual environment:

```bash
# 1. Clone the repository
git clone https://github.com/William-island/FaithDaIL.git
cd FaithDaIL

# 2. Install dependencies
pip install -r requirement.txt
```

*Note: Please ensure you have the correct versions of the [CARLA](https://carla.org/) and [MetaDrive](https://github.com/metadriverse/metadrive) simulators installed in your system before running the scripts.*

## 🚀 Quick Start

### 1. Training
You can start training models with the provided configuration files. The main script automatically loads parameters from `configs/`.

```bash
# Train on MetaDrive
python train_online.py --env_name metadrive --algo FaithDaIL

# Train on CARLA
python train_online.py --env_name carla --algo FaithDaIL
```

### 2. Evaluation
After training, evaluate your trained models using the evaluation scripts:
```bash
python eval_metadrive_models.py 
# or
python eval_carla_models.py
```

## 📝 Citation

If you find this code or our paper useful in your research, please consider citing:

```bibtex
@inproceedings{lingfaithful,
  title={Faithful Dynamic Imitation Learning from Human Intervention with Dynamic Regret Minimization},
  author={Ling, Bo and Gan, Zhengyu and Wang, Wanyuan and Gao, Guanyu and Wu, Weiwei and Lyu, Yan},
  booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems}
}
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Acknowledgments

- Based on the [MetaDrive](https://github.com/metadriverse/metadrive) and [CARLA](https://carla.org/) simulation platforms.
- Thanks to [DI-drive](https://github.com/opendilab/DI-drive) for their core autonomous driving environment utilities.

