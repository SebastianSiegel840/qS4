#!/bin/bash
#SBATCH --partition=pgi14
#SBATCH --nodes=1
#SBATCH --gpus=8

# i = 0

# for ql in 128 64 32 16 8 4 2 1
# do
#     python training.py --dataset cifar10 --grayscale --gpu $i --all_quant $ql --check_path all_quant_max_gr_S4legs &
#     echo "Running quantization sweep for $ql on GPU $i"
#     ((i += 1))
# done

python training.py --dataset cifar10 --grayscale --gpu 0 --all_quant 128 --check_path all_quant_surGrad_sine &
python training.py --dataset cifar10 --grayscale --gpu 1 --all_quant 64 --check_path all_quant_surGrad_sine &
python training.py --dataset cifar10 --grayscale --gpu 2 --all_quant 32 --check_path all_quant_surGrad_sine &
python training.py --dataset cifar10 --grayscale --gpu 3 --all_quant 16 --check_path all_quant_surGrad_sine &
python training.py --dataset cifar10 --grayscale --gpu 4 --all_quant 8 --check_path all_quant_surGrad_sine &
python training.py --dataset cifar10 --grayscale --gpu 5 --all_quant 4 --check_path all_quant_surGrad_sine &
python training.py --dataset cifar10 --grayscale --gpu 6 --all_quant 2 --check_path all_quant_surGrad_sine &
python training.py --dataset cifar10 --grayscale --gpu 7 --all_quant 1 --check_path all_quant_surGrad_sine &

wait