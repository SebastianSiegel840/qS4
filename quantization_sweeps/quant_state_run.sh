#!/bin/bash

for ql in 32 16 8 4 2 1
do
    python training.py --dataset cifar10 --gpu 7 --state_quant $ql --check_path state_quant_max_gr_S4legs --grayscale
done

