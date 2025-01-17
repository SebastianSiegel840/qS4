#!/bin/bash

for ql in 512 256 128 64 32 16 8 4 2 1
do
    python training.py --dataset cifar10 --grayscale --gpu 2 --all_quant $ql --check_path all_quant_max_gr_S4legs --lr 0.001
done