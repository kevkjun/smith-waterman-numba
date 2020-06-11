# Speeding up Smith-Waterman using Numba

Kevin Jun - MPCS 56420

## Executive Summary

The Smith-Waterman (SW) algorithm is a local sequence aligning DP algorithm following the recurrence relation

```math
res[i,j] = max(res[i-1, j-1] + s,
               res[i-1, j] + gap,
               res[i, j-1] + gap,
               0)
```

where `s` is a score derived from the residues corresponding to the `i` and `j` indices in a scoring matrix. SW suffers from poor performance when run at scale because the runtime is roughly polynomial - `O(n*m*l)` where `n` and `m` are the lengths of the two sequences being aligned and `l` is the size of the database. My project was to speed SW up using Python's `numba` package which provides just-in-time (JIT) compilation to compile functions to machine code at runtime, optimizing loops and computations on `numpy` arrays. Numba will run natively on CPU, but it also offers a CUDA API for GPU implementations. I compared runtimes for a serial SW implementation, a CPU Numba implementation, and a CUDA Numba implementation against a ~500 sequence excerpt of the `pdbaa` sequence database. Unsurprisingly, CPU Numba ran much faster than Serial and CUDA ran much faster than CPU. CPU Numba resulted in a `272x` speedup over Serial, and the fastest CUDA run `<grid_dim=8, block_dim=64>` was a `8,457x` speedup over CPU Numba and a `2,304,163x` speedup over Serial.

## Introduction

The Smith-Waterman (SW) algorithm is accurate and thorough for sequence alignment, but it suffers from poor performance at scale due to its polynomial runtime. Being able to run an SW alignment at scale in reasonable time would be beneficial in the beginning stages of research on a sequence in order to identify like-proteins. Though this utility is lessened considering blastp can be run reasonably quickly, it can be a useful step in the process. I will be presenting a faster SW implementation using Numba's JIT compilation on both CPU and GPU (CUDA).

## Background

There has been a long history of attempts to speed up the SW algorithm using advancements in computer architecture and engineering. There are many implementations using SIMD architectures which utilize vectorization for speedup; this is similar to Numba's CPU implementation. QIAGEN - a medical technology and research company - is currently the only company in bioinformatics to offer both SSE-vectorized (a SIMD Intel processor instruction set) and FPGA solutions, achieving speed-ups of more than 110 over other standard implementations. The fastest implementation on CPUs is through the open-source SWIPE software, capable of comparing residues from sixteen different database sequences. A speed of 106 billion cell updates per second was achieved on a dual Intel Xeon X5650 six-core processor system, which is faster than BLAST using the BLOSUM50 matrix.

## Methodology/Approach

The CPU and GPU Numba implementations required a refactoring of my original serial code because Numba compiles Python functions, not entire applications. I needed to decouple the alignment portion of my code from the rest, and I chose to do this by ditching the class structure that I'd used for the serial implementation and sticking with only two functions: `align` and `score`. `align` is called for each sequence alignment, and `score` is a sub-function called within `align` to score each cell. Because Numba is numpy-aware and especially benefits from loop vectorization, I changed any list comprehensions and dictionaries to nested loops and numpy arrays.

The most challenging part of coding for Numba was typing because Numba doesn't support many non-primitive Python data types like tuples or strings. Because my serial implementation used numpy arrays of chars representing sequences (e.g. `[A, B, E, D, K, H]`), I changed the chars to their respective ASCII codes (e.g. `A = 65`) so scores could be looked up in the score matrix by subtracting `65` (`A`) from the residue's ASCII code. The scoring matrix was implemented as a 2-D numpy array with the rows and columns associated with characters in alphabetical order (e.g. the score for residues `A` and `B` would be found at `matrix[0][1]`).

Numba can be sped up by using a `nopython` mode which indicates to the compiler that the decorated function can be run entirely without the involvement of the Python interpreter. To do this, I created and passed empty numpy arrays to the function to be used as the return and scratch computation arrays to avoid object instantiation inside the function. This also proved to be a boon when coding the CUDA implementation because CUDA requires that all arrays be allocated on the device prior to kernel launch. Furthermore, because allocating a full-sized scratch matrix of size `n*m` for the alignment sequence of length `n` and database sequence of length `m` for every sequence in the database is ridiculously costly, I allocated a scratch matrix of size `n*l` where `l` is the number of sequences in the database. As each residue of the `i`th database sequence is compared to the length of the alignment sequence, the cells in row `scratch[i]` are reused to calculate subsequent cells. This required keeping track of the `north` and `northwest` cells between iterations because this SW implementation used the high-road methodology.

Kernel launch was done using enough threads to allocate one thread for each sequence in the database. Thread count per block (`blockDim`) is passed as a command line argument `t_count`, and the block count (`gridDim`) is calculated accordingly.

## Results

The project worked and performed as I expected. The timing methodology was interesting because documentation says that the first run of a function compiled using Numba shouldn't be timed because the time will include compilation time. After the first run, the compiled code will be cached so subsequent calls will be much faster. The times below were from subsequent calls to the Numba-compiled functions so the true runtime could be measured.

The results matched closely to what I expected. The CPU Numba ran much faster than Serial, and CUDA ran much faster than the CPU Numba. The CUDA runtimes in the chart below correspond to kernel launches of `<grid_dim, block_dim>`.

CPU Numba represented a `272x` speedup over the serial implementation, and the fastest CUDA run `<grid_dim=8, block_dim=64>` was a `8,457x` speedup over CPU Numba and a `2,304,163x` speedup over Serial. Without a count of the registers used, I couldn't calculate the GPU occupancy so optimizing the `grid_dim` and `block_dim` was kind of a shot in the dark. In fact, I saw that the "best" `grid_dim` and `block_dim` combination swapped frequently.

| Implementation | Runtime (seconds) |
| :-------------| -------:|
| Serial         | 853.69264108 |
| CPU Numba      | 3.13341230 |
| CUDA <16,32>    | 0.0004447 |
| CUDA <8,64>     | 0.0003705 |
| CUDA <4,128>    | 0.0003776 |
| CUDA <2,256>    | 0.0003848 |

## Discussion

I'm not sure if this will help the field because there are other much faster implementations of SW available both commercially and as open source. However, this was a good case study in understanding how to utilize HPC and CUDA to achieve speedups for algorithms that have "time floors" like SW and other DP algos.

I would have liked to test this implementation against the entire pdbaa database. I was limited to only ~500 sequences because `Bio.SeqIO` does not error-check FASTA files for sequence validity so I had to hand-fix all of those sequences. I think there would be an even greater disparity in runtime between serial and CPU/GPU Numba with a larger database. Additionally, having a larger database would allow me to test the GPU saturation more. The number of threads (and by extension the `gridDim x blockDim` inputs) is capped by the number of sequences in the database so having a larger database would allow me to scale the algo and test more rigorously. I couldn't find a way to get compilation information from Numba meaning I couldn't figure out the number of registers used so I could plug it into the CUDA Occupancy Calculator and optimize GPU saturation.

There may have been an opportunity to create a shared array for the entire device to make data accesses faster for threads. I would have also liked to test the implementation with multi-dimensional grid and block kernel launches though this would only be necessary if there was a better implementation other than assigning one thread to each sequence.  Is there a way to have more than one thread working on a sequence at once?

## References

* “CUDA Programming.” Introduction to Numba: CUDA Programming, nyu-cds.github.io/python-numba/05-cuda/.
* Harris, Mark, et al. “Numba: High-Performance Python with CUDA Acceleration.” NVIDIA Developer Blog, 29 Apr. 2020, devblogs.nvidia.com/numba-python-cuda-acceleration/.
* Harrism. “Harrism/numba_examples.” GitHub, github.com/harrism/numba_examples/blob/master/mandelbrot_numba.ipynb.
* “Notes on Literal Types.” Notes on Literal Types - Numba 0.50.0.dev0+236.g64fbf2b-py3.7-Linux-x86_64.Egg Documentation, numba.pydata.org/numba-doc/dev/developer/literal.html.
* “Supported Python Features in CUDA Python.” Supported Python Features in CUDA Python - Numba 0.50.0.dev0+236.g64fbf2b-py3.7-Linux-x86_64.Egg Documentation, numba.pydata.org/numba-doc/dev/cuda/cudapysupported.html.
* “Writing CUDA Kernels.” Writing CUDA Kernels - Numba 0.50.0.dev0+236.g64fbf2b-py3.7-Linux-x86_64.Egg Documentation, numba.pydata.org/numba-doc/dev/cuda/kernels.html.
* “A ~5 Minute Guide to Numba.” A ~5 Minute Guide to Numba - Numba 0.49.1-py3.6-Macosx-10.7-x86_64.Egg Documentation, numba.pydata.org/numba-doc/latest/user/5minguide.html.
