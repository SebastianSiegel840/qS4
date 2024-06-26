#!/bin/bash

for ql in 16384 4096 1024 512 256 128 64 32 16 8 4 2 1
do
    python training.py --dataset cifar10 --grayscale --gpu 5 --all_quant $ql --act_quant 4 --check_path weight_quant_act2_max 
done

