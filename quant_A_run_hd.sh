#!/bin/bash

python training.py --dataset hd --gpu 7 --A_quant 16384 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 4096 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 1024 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 512 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 256 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 128 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 64 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 32 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 16 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 8 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 4 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 2 --check_path A_quant_max_hd_S4 --subsample 8
python training.py --dataset hd --gpu 7 --A_quant 1 --check_path A_quant_max_hd_S4 --subsample 8
