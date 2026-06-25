# Quantum Kernel Gaussian Process Regression with PennyLane

This repository implements a Quantum Kernel Gaussian Process Regression (QK-GPR) framework using PennyLane, PyTorch, and GPyTorch.

The project explores parameterized quantum feature maps as trainable kernels for Gaussian Process Regression and evaluates their performance on synthetic datasets.

## Features

* Quantum kernel implementation with PennyLane
* Gaussian Process Regression using GPyTorch
* Configurable number of qubits and variational layers
* Hyperparameter sweeps over:

  * Number of qubits
  * Number of layers
  * Encoding scales
* Experiment logging to Parquet files
* Automatic regression plot generation
* Reproducible experiments via fixed random seeds

## Project Structure

```text
quantum-kernel-gpr-pennylane/
│
├── src/
│   ├── data.py
│   ├── logger.py
│   ├── model.py
│   ├── plotting.py
│   ├── quantum_kernel.py
│   └── training.py
│
├── experiments/
│
├── plots/
│
├── quantum_kernel_gpr_pennylane_main_multiple.py
├── requirements.txt
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/finkalissa303/quantum-kernel-gpr-pennylane.git
cd quantum-kernel-gpr-pennylane
```

Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running Experiments

Run the experiment sweep:

```bash
python quantum_kernel_gpr_pennylane_main_multiple.py
```

Results will be saved to:

* `experiments/`
* `plots/`

## Configuration

The main script exposes the following experiment parameters:

* `QUBITS_LIST`
* `LAYERS_LIST`
* `SCALES_LIST`
* `epochs`
* `learning rate`
* `noise level`

These can be modified directly in the script.

## Background

Quantum kernel methods combine quantum feature maps with classical kernel learning techniques such as Gaussian Processes. PennyLane provides tools for constructing differentiable quantum circuits and integrating them with PyTorch-based machine learning workflows.

## License

MIT License
