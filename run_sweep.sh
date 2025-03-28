#!/bin/bash

sweep_id=frvhd7g2

CUDA_VISIBLE_DEVICES=0, wandb agent s-siegel/qSSM/$sweep_id &> logs/$sweep_id-00.log &

CUDA_VISIBLE_DEVICES=1, wandb agent s-siegel/qSSM/$sweep_id &> logs/$sweep_id-10.log &

CUDA_VISIBLE_DEVICES=2, wandb agent s-siegel/qSSM/$sweep_id &> logs/$sweep_id-20.log &

CUDA_VISIBLE_DEVICES=3, wandb agent s-siegel/qSSM/$sweep_id &> logs/$sweep_id-30.log &

CUDA_VISIBLE_DEVICES=4, wandb agent s-siegel/qSSM/$sweep_id &> logs/$sweep_id-40.log &

CUDA_VISIBLE_DEVICES=5, wandb agent s-siegel/qSSM/$sweep_id &> logs/$sweep_id-50.log &

CUDA_VISIBLE_DEVICES=6, wandb agent s-siegel/qSSM/$sweep_id &> logs/$sweep_id-60.log &

CUDA_VISIBLE_DEVICES=7, wandb agent s-siegel/qSSM/$sweep_id &> logs/$sweep_id-70.log &
wait