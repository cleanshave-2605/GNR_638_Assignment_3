# iResNet-CIFAR: Improved Residual Networks from Scratch

This repository contains a PyTorch implementation of **iResNet (Improved ResNet)** built from scratch and evaluated on the CIFAR-10 and CIFAR-100 datasets. This project implements the architecture proposed by Duta et al. in the paper [*"Improved Residual Networks for Image and Video Recognition"*](https://arxiv.org/abs/2004.04989) (CVPR 2020).

## Project Structure
There are two folders inside this repository - 
- `offical_implementation` consists of the official GitHub implementation of iResNet.
- `scratch_implementation` consists of the implementation of iResNet from scratch.

The plots for CIFAR-10 and CIFAR-100 are also given. Finally, the ```Report.pdf``` contains the report pertaining to the assignment.

## Installation

### To run the official code
First we create an environment to run the code
```bash
conda env create -f requirements.yml
```

Then, navigate to the ```offical_implementation``` folder and run the following code.
```bash
python main.py --dataset cifar10 --arch iresnet --model_depth 50 --n_classes 10 --batch-size 128 --lr 0.1
```

The above code will train and validate on CIFAR-10. To perform the same for CIFAR-100, run the following code.
```bash
python main.py --dataset cifar100 --arch iresnet --model_depth 50 --n_classes 100 --batch-size 128 --lr 0.1
```

### To run the implementation from scratch
Navigate to the ```scratch_implementation``` folder and run the following code.
```bash
bash ./cifar_10.sh
```

This will train and validate the model coded from scratch on CIFAR-10. To perform the same for CIFAR-100, run the following code.
```bash
bash ./cifar_100.sh
```

By changing the hyperparameter values in the shell files, different scenarios can be implemented.

## Results
The accuracy plots and log are stored in the ```results``` folder in both -- official and scratch implementation.