"""
Each thread gets a separate sequence to align.
usage: alignment.py [-h] --infile INFILE --scope {global,local} --gap
                    GAP --db DB --matrix {PAM250,BLOSUM62}
                    [--outfile OUTFILE]
"""

import numpy as np
import argparse

from numba import jit, cuda, types
from timeit import default_timer as timer

from Bio import SeqIO
from matrices import matrices


# ##################################################################
# ##################### CPU JIT IMPLEMENTATION #####################
# ##################################################################

@jit(nopython=True)
def score(i, j, gap, northwest, up, matrix, seq_res, db_seq_res, scratch):
    s = matrix[seq_res][db_seq_res]
    left = scratch[i][j-1] if j > 0 else 0
    
    return max(northwest + s, left + gap, up + gap, 0)


@jit(nopython=True)
def align(gap, matrix, seq, db, res, scratch):
    ref_seq_len = len(seq)
    # sequences are provided as list of ints (ASCII codes for residue char)
    for i, db_seq in enumerate(db):
        up = 0
        for j, seq_res in enumerate(seq):
            northwest = 0
            for k, db_seq_res in enumerate(db_seq):
                if db_seq_res == 0:
                    break
                new_northwest = scratch[i][j]
                # subtract by 65 bc 65 is ASCII code for 'A' - subtraction creates indices into scoring matrix
                scratch[i][j] = score(i, j, gap, northwest, up, matrix, seq_res - 65, db_seq_res - 65, scratch)
                northwest = new_northwest
                up = scratch[i][j+1 % len(seq)]
        res[i] = scratch[i][ref_seq_len - 1]
    return res


# ###############################################################
# ##################### CUDA IMPLEMENTATION #####################
# ###############################################################

# Define CUDA implementation for the score algorithm above
score_gpu = cuda.jit(device=True)(score)

@cuda.jit
def align_gpu(gap, matrix, seq, db, res, scratch):
    ref_seq_len = len(seq)
    up = 0
    thread_id = cuda.grid()
    for j, seq_res in enumerate(seq):
        northwest = 0
        for k, db_seq_res in enumerate(db[thread_id]):
                if db_seq_res == 0:
                    break
                new_northwest = scratch[thread_id][j]
                # subtract by 65 bc 65 is ASCII code for 'A' - subtraction creates indices into scoring matrix
                scratch[i][j] = score_gpu(thread_id, j, gap, northwest, up, matrix, seq_res - 65, db_seq_res - 65, scratch)
                northwest = new_northwest
                up = scratch[thread_id][j+1 % len(seq)]
        res[thread_id] = scratch[thread_id][ref_seq_len - 1]
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--scope', type=str, choices = ['global', 'local'], required=True)
    parser.add_argument('--gap', type=int, required=True)
    parser.add_argument('--db', type=str, required=True)
    parser.add_argument('--matrix', type=str, choices=['PAM250', 'BLOSUM62'], required=True)
    parser.add_argument('--impl', type=str, required=True, choices=['jit', 'cuda'])
    parser.add_argument('--outfile', type=str, default='align.txt')
    parser.add_argument('--t_count', type=int)

    args = parser.parse_args()
    
    with open(args.outfile, "w") as f:
        for record in SeqIO.parse(args.infile, "fasta"):
            header, alignment_seq = record.description, record.seq

        db = args.db
        scope = args.scope
        gap = -args.gap
        matrix = matrices.b62 if args.matrix == 'BLOSUM62' else matrices.p250
        impl = args.impl

        # implementation is incredibly ugly because converting a list of lists to a 2D numpy array requires the lists be of same length
        # so need to keep track of the longest seq length in the database and take max of that and the sequence to be aligned
        # and set that as the size of the inner list of the list of lists of db sequences
        db_seqs = []
        db_headers = []
        longest_seq_length = 0
        for record in SeqIO.parse(db, "fasta"):
            if len(record.seq) > longest_seq_length:
                longest_seq_length = len(record.seq)
            db_seqs.append(record.seq)
            db_headers.append(record.description)
        longest_length = max(longest_seq_length, len(alignment_seq))

        # unpack the strings in db_seqs
        db_seqs_chars = [[ord(db_seq[i]) if i < len(db_seq) else 0 for i in range(longest_length)] for db_seq in db_seqs]
        # create an np.array of np.array of ASCII codes for chars in sequences
        np_db_seqs = np.array(db_seqs_chars)

        f.write(f'Query Sequence:\n>{alignment_seq}\n\nDatabase: {db}\n\n')

        ####### Define dimensions for CUDA kernel #######
        if impl == 'cuda':
            block_dim = args.t_count
            grid_dim = len(db_seqs)//64 if len(db_seqs) % 64 == 0 else len(db_seqs)//64 + 1
        
        ####### Alignment #######
        start = timer() 

        # change alignment_seq to np.array of ASCII codes
        np_alignment_seq = np.array([ord(char) for char in str(alignment_seq)])
        # create scratch np.array with enough spaces to compute scores for each residue of the alignment sequence
        scratch = np.zeros((len(db_seqs), len(np_alignment_seq)), dtype=np.int64)
        # create the return np.array
        scores = np.zeros(len(db_seqs), dtype=np.int64)

        if impl == 'jit':
            scores = align(gap, matrix, np_alignment_seq, np_db_seqs, scores, scratch)
        else:
            scores = align_gpu[grid_dim, block_dim](gap, matrix, np_alignment_seq, db_seqs, scores, scratch)

        stop = timer()
        runtime = stop - start

        f.write(f'Implementation: {impl}\nRuntime: {runtime}\n\n')

        scores_list = list(scores)
        for score, desc in zip(scores_list, db_headers):
            f.write(f'> {score} | {desc}\n')

    