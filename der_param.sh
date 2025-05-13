#!/bin/bash
#SBATCH --partition=pgi14
#SBATCH --nodes=1
#SBATCH --gpus=4

#python param_2der.py --all_quant 128 --param_part "s4_layers.0" --gpu 0 --p_ckpt "for_prune_0/all128.pth" &
#python param_2der.py --all_quant 128 --param_part "s4_layers.1" --gpu 1 --p_ckpt "for_prune_0/all128.pth" &
#python param_2der.py --all_quant 128 --param_part "s4_layers.2" --gpu 2 --p_ckpt "for_prune_0/all128.pth" &
#python param_2der.py --all_quant 128 --param_part "s4_layers.3" --gpu 3 --p_ckpt "for_prune_0/all128.pth" &

#wait

# python param_2der.py --all_quant 1024 --param_part "s4_layers.0" --gpu 0 --p_ckpt "for_prune_0/all1024.pth" &
# python param_2der.py --all_quant 1024 --param_part "s4_layers.1" --gpu 1 --p_ckpt "for_prune_0/all1024.pth" &
# python param_2der.py --all_quant 1024 --param_part "s4_layers.2" --gpu 2 --p_ckpt "for_prune_0/all1024.pth" &
# python param_2der.py --all_quant 1024 --param_part "s4_layers.3" --gpu 3 --p_ckpt "for_prune_0/all1024.pth" &

# wait

python param_2der.py --param_part "s4_layers.0" --gpu 0 --p_ckpt "for_prune_0/baseline.pth" &
python param_2der.py --param_part "s4_layers.1" --gpu 1 --p_ckpt "for_prune_0/baseline.pth" &
python param_2der.py --param_part "s4_layers.2" --gpu 2 --p_ckpt "for_prune_0/baseline.pth" &
python param_2der.py --param_part "s4_layers.3" --gpu 3 --p_ckpt "for_prune_0/baseline.pth" &

wait