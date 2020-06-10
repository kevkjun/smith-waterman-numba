# Smith-Waterman Algo CUDA Implementation  
Kevin Jun - MPCS 56420

## Executive Summary

The Smith-Waterman algorithm is a local sequence aligning algorithm following the recurrence relation

```
res[i,j] = max(res[i-1, j-1] + s,
               res[i-1, j] + gap,
               res[i, j-1] + gap,
               0)
```

where s is a score derived from the residues corresponding to the `i` and `j` indices when looked up in a scoring matrix. Due to the nature of this algorithm, it suffers from poor performance when run at scale because the runtime is roughly polynomial - O(n*m) = O($n^2$) where `n` and `m` are the lengths of the two sequences being aligned. My project looked to speed this up using Python's `numba` package which provides just-in-time (JIT) compilation to compile functions to optimized machine code at runtime, offering optimization for loop-based computations and `numpy` arrays. Numba runs natively on CPUs, but also offers a CUDA API for further computation speedups. I compared runtimes for a serial SW implementation, a CPU Numba implementation, and a CUDA implementaiton using Numba against a ~500 sequence excerpt of the `pdbaa` sequence database.

## Introduction

Though the Smith-Waterman (SW) algorithm is accurate and thorough, it suffers from poor performance at scale due to its `O(n*m)` runtime. Being able to run an SW alignment at scale in reasonable time would be beneficial in the beginning stages of research on a sequence in order to identify like proteins. Though this utility is lessened considering BLAST can be run reasonably quickly, it can be a useful step in the process. I will be presenting a faster SW implementation using Numba's JIT compilation on both CPU and GPU (CUDA).

## Background

## Methodology/Approach

## Results

## Discussion

## References