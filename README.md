# README - MPCS 56420 Final Project

## Requirements

### Packages

* biopython
* numpy
* numba
* cudatoolkit

### System

* CUDA-enabled GPU
* CUDA Toolkit (at least version 8.0)

## Running the code

### `align.py`

This is the standard serial version of the sequence alignment code. Function signature is as follows:

```console
python align.py --infile ../data/spike.fasta --scope local --gap 3 --db ../data/spike.fasta --matrix BLOSUM62 --outfile ../results/align_serial.txt
```

### `align_numba.py`

#### JIT CPU

Uses Numba's JIT compilation to compile alignment function to machine code. Function signature is as follows:

```console
python align_numba.py --infile ../data/spike.fasta  --gap 5 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --outfile ../results/align_jit.txt --impl jit
```

#### CUDA

Uses Numba's CUDA API to launch kernel that assigns a thread to align the input sequence with each sequence in the database. Function signature is as follows:

```console
python align_numba.py --infile ../data/spike.fasta  --gap 5 --db ../data/excerpt_pdbaa.fasta --matrix BLOSUM62 --outfile ../results/align_cuda.txt --impl cuda --t_count 64
```

For better performance, keep the `t_count` (thread count) to a multiple of 32.

### Resources

https://github.com/harrism/numba_examples/blob/master/mandelbrot_numba.ipynb
https://numba.pydata.org/numba-doc/latest/user/5minguide.html
https://numba.pydata.org/numba-doc/dev/cuda/cudapysupported.html
https://numba.pydata.org/numba-doc/dev/cuda/kernels.html#kernel-declaration
https://nyu-cds.github.io/python-numba/05-cuda/
https://devblogs.nvidia.com/numba-python-cuda-acceleration/
https://numba.pydata.org/numba-doc/dev/developer/literal.html
