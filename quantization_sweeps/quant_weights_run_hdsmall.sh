#!/bin/bash

for ql in 256 128 64 32 16 8 4 2
do
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 28 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state28 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 28 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state28 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 28 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state28 --nonlin relu
done

for ql in 256 128 64 32 16 8 4 2
do
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 14 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state14 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 14 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state14 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 14 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state14 --nonlin relu
done

for ql in 256 128 64 32 16 8 4 2
do
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 7 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state7 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 7 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state7 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 7 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state7 --nonlin relu
done

for ql in 256 128 64 32 16 8 4 2
do
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 4 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state4 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 4 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state4 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 4 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state4 --nonlin relu
done

for ql in 256 128 64 32 16 8 4 2
do
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 3 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state3 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 3 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state3 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 3 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state3 --nonlin relu
done

for ql in 256 128 64 32 16 8 4 2
do
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 2 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state2 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 2 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state2 --nonlin relu
    python training.py --dataset hd --hd_small --subsample 64 --n_layers_m 1 --d_model_m 3 --d_state 2 --all_quant $ql --state_quant 256 --act_quant 256 --check_path hd_small_quant_state2 --nonlin relu
done
