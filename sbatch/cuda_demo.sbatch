#!/bin/bash
#SBATCH --job-name=cuda_demo
#SBATCH --output=cuda_demo.log
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1

module load cuda/8.0
module load Anaconda3
source activate mpcs56420

printf "block_dim = 32\n"
python ../src/align_numba.py --infile ../data/spike.fasta --gap 3 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --out ../results/cuda_demo.txt --impl cuda --t_count 32