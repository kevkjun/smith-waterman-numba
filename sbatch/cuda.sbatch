#!/bin/bash
#SBATCH --job-name=align_cuda
#SBATCH --output=align_cuda.out
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:1

module load cuda/8.0
module load Anaconda3
source activate mpcs56420

printf "block_dim = 32\n"
python ../src/align_numba.py --infile ../data/spike.fasta --gap 3 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --out ../results/align_cuda_32.txt --impl cuda --t_count 32

printf "block_dim = 64\n"
python ../src/align_numba.py --infile ../data/spike.fasta --gap 3 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --out ../results/align_cuda_64.txt --impl cuda --t_count 64

printf "block_dim = 128\n"
python ../src/align_numba.py --infile ../data/spike.fasta --gap 3 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --out ../results/align_cuda_128.txt --impl cuda --t_count 128

printf "block_dim = 256\n"
python ../src/align_numba.py --infile ../data/spike.fasta --gap 3 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --out ../results/align_cuda_256.txt --impl cuda --t_count 256