#!/bin/bash

for ql in 16384 4096 1024 512 256 128 64 32 16 8 4 2 1
do
    python training.py --dataset cifar10 --gpu 0 --all_quant ql --act_quant 256 --check_path weight_quant_act8_max --grayscale
done

#python training.py --dataset cifar10 --gpu 0 --all_quant 16384 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 4096 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 1024 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 512 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 256 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 128 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 64 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 32 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 16 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 8 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 4 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 2 --check_path weight_quant_act8_max --grayscale
#python training.py --dataset cifar10 --gpu 0 --all_quant 1 --check_path weight_quant_act8_max --grayscale
