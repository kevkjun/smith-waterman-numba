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

### Numba JIT CPU implementation


