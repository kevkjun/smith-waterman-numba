#!/bin/bash
#SBATCH --job-name=multi_alignment_eight           # Job name
#SBATCH --mail-type=ALL               # Mail events (NONE, BEGIN, END, FAIL, ALL)
#SBATCH --mail-user=kjun@uchicago.edu    # Where to send mail
#SBATCH --ntasks=8                    # Run on a single CPU
#SBATCH --cpus-per-task=1                # Number of threads per task
#SBATCH --mem=32gb                     # Job memory request
#SBATCH --time=05:00:00               # Time limit hrs:min:sec
#SBATCH --output=multi_alignment_eight.out

module load Anaconda3
source activate mpcs56420

python multi_alignment.py --infile spike.fasta --scope global --gap 3 --db /project2/mpcs56420/db/pdbaa.fasta --matrix BLOSUM62 --out multi_align.txt --ncores 8