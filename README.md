# PINN for freeze-drying

This project models the freeze-drying kinetics of fruits using Physics-Informed Neural Networks (PINNs). The model learns the relationship between time, thickness, and chamber pressure with residual moisture (MR) and effective diffusivity (Deff).

## Overview

1. The PINN architecture was implemented in PyTorch
2. Physical loss terms based on Fick's Second Law to regularize the solution
3. Experimental data for citron (Citrus medica) and strawberry (Fragaria x ananassa

## Main objectives

- predict residual moisture (MR) over time
- infer the effective diffusivity (Deff) of the material
- infer the drying time across different thickness and pressure condition

## Project structure

- `main.py`: main model training script
- `simulate.py`: custom scenario execution
- `requirements.txt`: project dependencies
- `data/`: experimental datasets (`citrus_data.csv`, `strawberry_data.csv`)
- `files/configs.yaml`: model hyperparameters and physical limits
- `files/models/`: saved trained weights in `.pt` format
- `src/pinn/`: model, loss, and training implementation
- `src/visualizer/`: plotting and validation functions
- `src/utils.py`: utilities and configuration loader

## Requirements

- Python 3.10+

```bash
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

## How to use

### 1) Train the model

```bash
python main.py
```

This script:

- loads the configuration from `files/configs.yaml`
- reads the input data
- builds the physical grid for the problem
- trains the PINN
- saves the model in `files/models/`
- runs the initial evaluation

### 2) Evaluate a saved model

Evaluates the quality of the PINN's fit to the experimental data.

```bash
python -m src.pinn.evaluate
```

### 3) Simulate a custom scenario

Simulates the drying time required to reach a target moisture ratio under varying chamber pressure and material thickness conditions, in order to evaluate the model's interpolation and extrapolation capabilities.

```bash
python simulate.py
```

## Configuration

The main settings are in `files/configs.yaml`, including:

- model normalization limits
- number of points in the physical grid
- training epochs
- learning rate
- loss term weights

## Notes

- The project was developed for fruit datasets such as citrus and strawberry.
- Trained models are already available in `files/models/`.
- You can adjust `main.py` and `simulate.py` to change the dataset, architecture, or simulation scenario.



For any further information, feel free to email me at kleberson.alves@ufape.edu.br or klebersonfa18@gmail.com
