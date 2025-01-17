#!/bin/bash

for ql in 16384 4096 1024 512 256 128 64 32 16 8 4 2 1
do
    python training.py --dataset cifar10 --gpu 0 --A_quant $ql --check_path A_quant_max_gr_S4legs --grayscale --lr 0.001
done