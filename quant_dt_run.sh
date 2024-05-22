#!/bin/bash

python training.py --dataset cifar10 --gpu 3 --dt_quant 16384 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 4096 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 1024 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 512 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 256 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 128 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 64 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 32 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 16 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 8 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 4 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 2 --check_path dt_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 3 --dt_quant 1 --check_path dt_quant_max_S4 --grayscale
