#!/bin/bash

for ql in 0.002 0.004 0.006 0.008
do
    python training.py --dataset cifar10 --gpu 0 --A_quant 1024 --check_path lr_series_A1024 --grayscale --lr $ql
done