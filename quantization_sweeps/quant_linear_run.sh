#!/bin/bash

for ql in 256 128 64 32 16 8 4 2 1
do
    python training.py --dataset cifar10 --grayscale --gpu 3 --linear_quant $ql --check_path linear_quant_max_gr_S4fourier --measure fourier
done
