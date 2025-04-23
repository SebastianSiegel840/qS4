#!/bin/bash
#SBATCH --partition=pgi14
#SBATCH --nodes=1
#SBATCH --gpus=4

python param_wiggle.py --all_quant 128 --param_part "s4_layers.0" --gpu 0 &
python param_wiggle.py --all_quant 128 --param_part "s4_layers.1" --gpu 1 &
python param_wiggle.py --all_quant 128 --param_part "s4_layers.2" --gpu 2 &
python param_wiggle.py --all_quant 128 --param_part "s4_layers.3" --gpu 3 &

wait

python param_wiggle.py --all_quant 128 --param_part "norms.0" --gpu 0 &
python param_wiggle.py --all_quant 128 --param_part "norms.1" --gpu 1 &
python param_wiggle.py --all_quant 128 --param_part "norms.2" --gpu 2 &
python param_wiggle.py --all_quant 128 --param_part "norms.3" --gpu 3 &

wait

python param_wiggle.py --all_quant 128 --param_part "coder" --gpu 0 &