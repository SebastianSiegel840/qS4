#!/bin/bash

for ql in 0 5e-3 1e-2 1.5e-2 2e-2 2.5e-2 3e-2 3.5e-2 4e-2 4.5e-2 5e-2
do
    python training.py --dataset hd --subsample 8 --n_layers_m 1 --gpu 0  --weight_noise $ql --check_path A_noise_max_hd_S4D_1l_noise
done
