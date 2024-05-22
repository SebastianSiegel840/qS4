#!/bin/bash

python training.py --dataset hd --gpu 3 --linear_quant 16384 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 4096 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 1024 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 512 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 256 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 128 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 64 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 32 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 16 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 8 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 4 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 2 --check_path linear_quant_max_hd_1l_64 --subsample 8
python training.py --dataset hd --gpu 3 --linear_quant 1 --check_path linear_quant_max_hd_1l_64 --subsample 8
