#!/bin/bash

python training.py --dataset cifar10 --gpu 1 --linear_quant 65536 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 16384 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 4096 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 1024 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 512 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 256 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 128 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 64 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 32 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 16 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 8 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 4 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 2 --check_path linear_quant_max_gr --grayscale
python training.py --dataset cifar10 --gpu 1 --linear_quant 1 --check_path linear_quant_max_gr --grayscale
