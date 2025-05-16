# Code For NeurIPS 2025
Welcome to the code repository for our NeurIPS 2025 submission!

## Overview

This repository contains the implementation, experiments, and supplementary materials for our paper submitted to NeurIPS 2025. The code is organized for easy reproduction of results and further research.

## Requirements

- Python 3.8+
- PyTorch >= 1.10
- NumPy

Install dependencies with:

```bash
pip install -r requirements.txt
```


## Installation

We recommend using `conda` to manage environments.

### MetaDrive
```bash
pip install metadrive==0.2.5
pip install -e .
```
### CARLA
```bash
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
tar -xf CARLA_0.9.10.1.tar.gz

# Set environment variables (adjust path as needed)
export CARLA_ROOT=~/CARLA_0.9.10.1
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg:$PYTHONPATH"

# Install dependencies
pip install DI-engine==0.2.2 markupsafe==2.0.1
```


## Usage

To run the main experiments:

```bash
python train_online.py
```

Modify the configuration files in the `configs/` directory to adjust experiment settings.

## Repository Structure

- `train_online.py` — Entry point for training and evaluation
- `algos/` — Model architectures
- `envs/` — Environment files
- `configs/` — Experiment configuration files for each simulator
- `logs/` — Output and logs



## License

This project is licensed under the MIT License.

