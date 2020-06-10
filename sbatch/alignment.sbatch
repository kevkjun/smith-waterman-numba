#!/bin/bash
#SBATCH --job-name=align           # Job name
#SBATCH --output=align.out
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=1                # Number of threads per task
#SBATCH --mem=32gb                     # Job memory request
#SBATCH --time=01:00:00               # Time limit hrs:min:sec

module load Anaconda3
source activate mpcs56420

python ../src/align.py --infile ../data/spike.fasta --scope local --gap 3 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --out ../results/align_serial.txt