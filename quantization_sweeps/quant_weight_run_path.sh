#!/bin/bash

for ql in 16 8 4 2 1 #16384 4096 1024 512 256 128 64 32 16 8 4 2 1
do
    python training.py --dataset pathfinder --lr 0.004 --weight_decay 0.03 --gpu 6 --all_quant $ql --act_quant None --check_path weight_quant_actNone_max_S4D_path
done