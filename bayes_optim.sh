#!/bin/bash
#SBATCH --job-name qSSM
#SBATCH --time 7-24:00:00
#SBATCH --signal=USR1
#SBATCH -p pgi14
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH 
#SBATCH --cpus-per-task=1
#SBATCH -o ./log/job_%A_%a.o
#SBATCH -e ./log/job_%A_%a.e
#SBATCH --mail-type=BEGIN,END,FAIL,REQUEUE
#SBATCH --mail-user=s.siegel@fz-juelich.de
srun run_sweep.sh
wait
scontrol show jobid ${SLURM_JOBID} -dd # Job summary at exit