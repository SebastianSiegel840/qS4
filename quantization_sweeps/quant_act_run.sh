#!/bin/bash

for ql in 16 8 4 2 1
do
    python training.py --dataset cifar10 --gpu 1 --act_quant $ql --check_path act_quant_max_gr_S4legs --grayscale
done
