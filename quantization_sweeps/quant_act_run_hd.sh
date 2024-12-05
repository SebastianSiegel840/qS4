#!/bin/bash
#!/bin/bash

for ql in 16384 4096 1024 512 256 128 64 32 16 8 4 2 1
do
    python training.py --dataset hd --subsample 8 --n_layers_m 1 --gpu 1  --act_quant $ql --check_path act_quant_max_hd_S4D_1l_noise
done
