#!/bin/bash

for ql in 16384 4096 1024 512 256 128 64 32 16 8 4 2 1
do
    python training.py --dataset hd --gpu 3 --state_quant $ql --check_path state_quant_max_hd_1l --subsample 8 --n_layers_m 1
done