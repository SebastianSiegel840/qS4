#!/bin/bash

python training.py --dataset cifar10 --gpu 5 --all_quant 16384 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 4096 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 1024 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 512 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 256 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 128 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 64 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 32 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 16 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 8 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 4 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 2 --check_path all_quant_max_S4 --grayscale
python training.py --dataset cifar10 --gpu 5 --all_quant 1 --check_path all_quant_max_S4 --grayscale
