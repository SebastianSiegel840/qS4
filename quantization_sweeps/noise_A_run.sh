#!/bin/bash

for ql in 0 5e-3 1e-2 1.5e-2 2e-2 2.5e-2 3e-2 3.5e-2 4e-2 4.5e-2 5e-2
do
    python training.py --dataset cifar10 --grayscale --gpu 1 --check_path A_noise_max_gr --weight_noise $ql
done