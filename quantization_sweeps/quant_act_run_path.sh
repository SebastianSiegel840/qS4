#!/bin/bash

for ql in 1 #16384 4096 1024 512 256 128 64 32 16 8 4 2 1
do
    python training.py --dataset pathfinder --lr 0.004 --weight_decay 0.03 --gpu 2 --act_quant $ql --check_path act_quant_max_S4D_path
done