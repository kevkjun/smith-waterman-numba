# Smith-Waterman Algo CUDA Implementation  
Kevin Jun - MPCS 56420

## Executive Summary

The Smith-Waterman algorithm is a local sequence aligning DP algorithm following the recurrence relation

```
res[i,j] = max(res[i-1, j-1] + s,
               res[i-1, j] + gap,
               res[i, j-1] + gap,
               0)
```

where `s` is a score derived from the residues corresponding to the `i` and `j` indices when looked up in a scoring matrix. Due to the nature of this algorithm, it suffers from poor performance when run at scale because the runtime is roughly polynomial - `O(n*m*l)` where `n` and `m` are the lengths of the two sequences being aligned and `l` is the size of the database. My project looked to speed this up using Python's `numba` package which provides just-in-time (JIT) compilation to compile functions to optimized machine code at runtime, offering optimization for loop-based computations and `numpy` arrays. Numba runs natively on CPUs, but also offers a CUDA API for further computation speedups. I compared runtimes for a serial SW implementation, a CPU Numba implementation, and a CUDA Numba implementation against a ~500 sequence excerpt of the `pdbaa` sequence database.

## Introduction

Though the Smith-Waterman (SW) algorithm is accurate and thorough, it suffers from poor performance at scale due to its polynomial runtime. Being able to run an SW alignment at scale in reasonable time would be beneficial in the beginning stages of research on a sequence in order to identify like-proteins. Though this utility is lessened considering blastp can be run reasonably quickly, it can be a useful step in the process. I will be presenting a faster SW implementation using Numba's JIT compilation on both CPU and GPU (CUDA).

## Background

There has been a long history of attempts to speed up the SW algorithm using advancements in computer architecture and engineering. There exist many implementations using SIMD architectures which utilize vectorization for speedup; this is the same methodology utilized by Numba through its JIT compilation and CPU execution. QIAGEN - a medical technology and research company - is currently the only company in bioinformatics to offer both SSE-vectorized (SIMD Intel processor instruction set) and FPGA solutions, achieving speed-ups of more than 110 over other standard implementations. The SWIPE software offers the fastest implementatio on CPUs, capable of comparing residues from sixteen different database sequences and achieving a speed of 106 billion cell updates per second which is faster than BLAST when using the BLOSUM50 matrix.

## Methodology/Approach

Writing the CPU and GPU Numba SW implementations required a refactoring of my original serial code because Numba compiles Python functions, not entire applications. I needed to decouple the alignment portion of my code from the rest, and I chose to do this by ditching the class structure that I'd used for the serial implementation and sticking with only two functions: `align` and `score`. `align` is called for each sequence alignment, and `score` is a sub-function called within `align` to score each cell. Because Numba is numpy-aware and especially benefits from loop vectorization, I refactored any list comprehensions and dictionaries to nested loops and numpy arrays.

I found the most challenging part in coding for Numba was typing because Numba doesn't support many Python data types like tuples or strings. Because my previous implementation had used numpy arrays of chars representing sequences (e.g. `[A, B, E, D, K, H]`), I needed to change these representations to their respective ASCII codes (e.g. `A = 65`) so scores could be looked up in the score matrix by subtracting `65` (`A`) from the ASCII code. The scoring matrix was implemented as a 2-D numpy array with the rows and columns associated with characters in alphabetical order (e.g. the score for residues `A` and `B` would be found at `matrix[0][1]`).

Numba can be sped up by using a `nopython` mode which indicates to the compiler that the decorated function can be run entirely without the involvement of the Python interpreter. To do this, I created and passed empty numpy arrays to the function to be used as the return and scratch computation arrays to avoid any object creation within the function. This also proved to be a boon when coding the CUDA implementation because CUDA requires that all arrays be allocated on the device prior to kernel launch. Furthermore, because allocating a full-sized scratch matrix of size `n*m` for alignment sequence of length `n` and database sequence of length `m` for every sequence in the database is ridiculously costly, I allocated a scratch matrix of size `n*l` where `l` is the number of sequences in the database. As each residue of the `i`th database sequence is compared to the length of the alignment sequence, the cells in row `scratch[i]` are reused to calculate subsequent cells. This required keeping track of the `north` and `northwest` cells between iterations because this SW implementation used the high-road methodology.

Kernel launch was done using enough threads to allocate one thread to each sequence in the database. Thread count per block (`blockDim`) is passed as a command line argument `t_count`, and the block count (`gridDim`) is determined by finding the blocks required to create enough threads.

## Results

## Discussion

I'm not sure if this will help the field because there are other much faster implementations of SW both available commercially and as open source. However, this was a good case study in understanding how to utilize HPC and CUDA to achieve speedups for algorithms that have "time floors" like SW and other DP algos.

With more time and resources, I would have liked to test this implementation against the entire pdbaa database. I was limited to only ~500 sequences because my implementation (as well as `BioPython.SeqIO`) does not error-check for FASTA file or sequence validity. I actually hand-fixed all of those sequences.

## References