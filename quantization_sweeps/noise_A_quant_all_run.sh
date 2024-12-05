#!/bin/bash

for ql in 16384 4096 1024 512 256 128 64 32 16
do
    for nl in 0.06 0.07 0.08 0.09 0.1 0.12 0.14 0.16 0.18 0.2 0.25 0.3
    do
        python training.py --dataset cifar10 --grayscale --gpu 2 --check_path A_noise_all_quant_max_gr --weight_noise $nl --all_quant $ql
    done
done
