#!/bin/bash

for ql in 16384 4096 1024 512 256 128 64 32 16 8 4 2 1
do
    python training.py --dataset hd --n_layers_m 1 --d_model_m 3 --gpu 6 --d_state 7 --subsample 64 --hd_small --nonlin relu --A_quant $ql --check_path A_quant_max_relu_hdsmall
done