#!/bin/bash
#SBATCH --job-name=align_jit           # Job name
#SBATCH --output=align_jit.out
#SBATCH --ntasks=1                    # Run on a single CPU
#SBATCH --cpus-per-task=1                # Number of threads per task
#SBATCH --mem=32gb                     # Job memory request
#SBATCH --time=00:10:00               # Time limit hrs:min:sec

module load Anaconda3
source activate mpcs56420

printf "JIT Run 1\n"
python ../src/align_numba.py --infile ../data/spike.fasta --gap 3 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --out ../results/align_jit.fasta --impl jit

printf "JIT Run 2\n"
python ../src/align_numba.py --infile ../data/spike.fasta --gap 3 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --out ../results/align_jit1.fasta --impl jit

printf "JIT Run 3\n"
python ../src/align_numba.py --infile ../data/spike.fasta --gap 3 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --out ../results/align_jit2.fasta --impl jit

printf "JIT Run 4\n"
python ../src/align_numba.py --infile ../data/spike.fasta --gap 3 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --out ../results/align_jit3.fasta --impl jit