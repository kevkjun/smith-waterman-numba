"""
usage: alignment.py [-h] --infile INFILE --scope {global,local} --gap
                    GAP --db DB --matrix {PAM250,BLOSUM62}
                    [--outfile OUTFILE]
"""

import numpy as np
import heapq as hp
import argparse
import timeit

# Midway requires lowercase bio but other machines require upper
try:
    from Bio import SeqIO
except:
    from bio import SeqIO

# found the matrix here: https://www.biostars.org/p/405990/
blosum62 = {
    ('W', 'F'): 1, ('L', 'R'): -2, ('S', 'P'): -1, ('V', 'T'): 0,
    ('Q', 'Q'): 5, ('N', 'A'): -2, ('Z', 'Y'): -2, ('W', 'R'): -3,
    ('Q', 'A'): -1, ('S', 'D'): 0, ('H', 'H'): 8, ('S', 'H'): -1,
    ('H', 'D'): -1, ('L', 'N'): -3, ('W', 'A'): -3, ('Y', 'M'): -1,
    ('G', 'R'): -2, ('Y', 'I'): -1, ('Y', 'E'): -2, ('B', 'Y'): -3,
    ('Y', 'A'): -2, ('V', 'D'): -3, ('B', 'S'): 0, ('Y', 'Y'): 7,
    ('G', 'N'): 0, ('E', 'C'): -4, ('Y', 'Q'): -1, ('Z', 'Z'): 4,
    ('V', 'A'): 0, ('C', 'C'): 9, ('M', 'R'): -1, ('V', 'E'): -2,
    ('T', 'N'): 0, ('P', 'P'): 7, ('V', 'I'): 3, ('V', 'S'): -2,
    ('Z', 'P'): -1, ('V', 'M'): 1, ('T', 'F'): -2, ('V', 'Q'): -2,
    ('K', 'K'): 5, ('P', 'D'): -1, ('I', 'H'): -3, ('I', 'D'): -3,
    ('T', 'R'): -1, ('P', 'L'): -3, ('K', 'G'): -2, ('M', 'N'): -2,
    ('P', 'H'): -2, ('F', 'Q'): -3, ('Z', 'G'): -2, ('X', 'L'): -1,
    ('T', 'M'): -1, ('Z', 'C'): -3, ('X', 'H'): -1, ('D', 'R'): -2,
    ('B', 'W'): -4, ('X', 'D'): -1, ('Z', 'K'): 1, ('F', 'A'): -2,
    ('Z', 'W'): -3, ('F', 'E'): -3, ('D', 'N'): 1, ('B', 'K'): 0,
    ('X', 'X'): -1, ('F', 'I'): 0, ('B', 'G'): -1, ('X', 'T'): 0,
    ('F', 'M'): 0, ('B', 'C'): -3, ('Z', 'I'): -3, ('Z', 'V'): -2,
    ('S', 'S'): 4, ('L', 'Q'): -2, ('W', 'E'): -3, ('Q', 'R'): 1,
    ('N', 'N'): 6, ('W', 'M'): -1, ('Q', 'C'): -3, ('W', 'I'): -3,
    ('S', 'C'): -1, ('L', 'A'): -1, ('S', 'G'): 0, ('L', 'E'): -3,
    ('W', 'Q'): -2, ('H', 'G'): -2, ('S', 'K'): 0, ('Q', 'N'): 0,
    ('N', 'R'): 0, ('H', 'C'): -3, ('Y', 'N'): -2, ('G', 'Q'): -2,
    ('Y', 'F'): 3, ('C', 'A'): 0, ('V', 'L'): 1, ('G', 'E'): -2,
    ('G', 'A'): 0, ('K', 'R'): 2, ('E', 'D'): 2, ('Y', 'R'): -2,
    ('M', 'Q'): 0, ('T', 'I'): -1, ('C', 'D'): -3, ('V', 'F'): -1,
    ('T', 'A'): 0, ('T', 'P'): -1, ('B', 'P'): -2, ('T', 'E'): -1,
    ('V', 'N'): -3, ('P', 'G'): -2, ('M', 'A'): -1, ('K', 'H'): -1,
    ('V', 'R'): -3, ('P', 'C'): -3, ('M', 'E'): -2, ('K', 'L'): -2,
    ('V', 'V'): 4, ('M', 'I'): 1, ('T', 'Q'): -1, ('I', 'G'): -4,
    ('P', 'K'): -1, ('M', 'M'): 5, ('K', 'D'): -1, ('I', 'C'): -1,
    ('Z', 'D'): 1, ('F', 'R'): -3, ('X', 'K'): -1, ('Q', 'D'): 0,
    ('X', 'G'): -1, ('Z', 'L'): -3, ('X', 'C'): -2, ('Z', 'H'): 0,
    ('B', 'L'): -4, ('B', 'H'): 0, ('F', 'F'): 6, ('X', 'W'): -2,
    ('B', 'D'): 4, ('D', 'A'): -2, ('S', 'L'): -2, ('X', 'S'): 0,
    ('F', 'N'): -3, ('S', 'R'): -1, ('W', 'D'): -4, ('V', 'Y'): -1,
    ('W', 'L'): -2, ('H', 'R'): 0, ('W', 'H'): -2, ('H', 'N'): 1,
    ('W', 'T'): -2, ('T', 'T'): 5, ('S', 'F'): -2, ('W', 'P'): -4,
    ('L', 'D'): -4, ('B', 'I'): -3, ('L', 'H'): -3, ('S', 'N'): 1,
    ('B', 'T'): -1, ('L', 'L'): 4, ('Y', 'K'): -2, ('E', 'Q'): 2,
    ('Y', 'G'): -3, ('Z', 'S'): 0, ('Y', 'C'): -2, ('G', 'D'): -1,
    ('B', 'V'): -3, ('E', 'A'): -1, ('Y', 'W'): 2, ('E', 'E'): 5,
    ('Y', 'S'): -2, ('C', 'N'): -3, ('V', 'C'): -1, ('T', 'H'): -2,
    ('P', 'R'): -2, ('V', 'G'): -3, ('T', 'L'): -1, ('V', 'K'): -2,
    ('K', 'Q'): 1, ('R', 'A'): -1, ('I', 'R'): -3, ('T', 'D'): -1,
    ('P', 'F'): -4, ('I', 'N'): -3, ('K', 'I'): -3, ('M', 'D'): -3,
    ('V', 'W'): -3, ('W', 'W'): 11, ('M', 'H'): -2, ('P', 'N'): -2,
    ('K', 'A'): -1, ('M', 'L'): 2, ('K', 'E'): 1, ('Z', 'E'): 4,
    ('X', 'N'): -1, ('Z', 'A'): -1, ('Z', 'M'): -1, ('X', 'F'): -1,
    ('K', 'C'): -3, ('B', 'Q'): 0, ('X', 'B'): -1, ('B', 'M'): -3,
    ('F', 'C'): -2, ('Z', 'Q'): 3, ('X', 'Z'): -1, ('F', 'G'): -3,
    ('B', 'E'): 1, ('X', 'V'): -1, ('F', 'K'): -3, ('B', 'A'): -2,
    ('X', 'R'): -1, ('D', 'D'): 6, ('W', 'G'): -2, ('Z', 'F'): -3,
    ('S', 'Q'): 0, ('W', 'C'): -2, ('W', 'K'): -3, ('H', 'Q'): 0,
    ('L', 'C'): -1, ('W', 'N'): -4, ('S', 'A'): 1, ('L', 'G'): -4,
    ('W', 'S'): -3, ('S', 'E'): 0, ('H', 'E'): 0, ('S', 'I'): -2,
    ('H', 'A'): -2, ('S', 'M'): -1, ('Y', 'L'): -1, ('Y', 'H'): 2,
    ('Y', 'D'): -3, ('E', 'R'): 0, ('X', 'P'): -2, ('G', 'G'): 6,
    ('G', 'C'): -3, ('E', 'N'): 0, ('Y', 'T'): -2, ('Y', 'P'): -3,
    ('T', 'K'): -1, ('A', 'A'): 4, ('P', 'Q'): -1, ('T', 'C'): -1,
    ('V', 'H'): -3, ('T', 'G'): -2, ('I', 'Q'): -3, ('Z', 'T'): -1,
    ('C', 'R'): -3, ('V', 'P'): -2, ('P', 'E'): -1, ('M', 'C'): -1,
    ('K', 'N'): 0, ('I', 'I'): 4, ('P', 'A'): -1, ('M', 'G'): -3,
    ('T', 'S'): 1, ('I', 'E'): -3, ('P', 'M'): -2, ('M', 'K'): -1,
    ('I', 'A'): -1, ('P', 'I'): -3, ('R', 'R'): 5, ('X', 'M'): -1,
    ('L', 'I'): 2, ('X', 'I'): -1, ('Z', 'B'): 1, ('X', 'E'): -1,
    ('Z', 'N'): 0, ('X', 'A'): 0, ('B', 'R'): -1, ('B', 'N'): 3,
    ('F', 'D'): -3, ('X', 'Y'): -1, ('Z', 'R'): 0, ('F', 'H'): -1,
    ('B', 'F'): -3, ('F', 'L'): 0, ('X', 'Q'): -1, ('B', 'B'): 4
}
# found the matrix here: http://www.cs.grinnell.edu/~rebelsky/ExBioPy/Code/pam250.txt
pam250 = {
    ('W', 'F'): 0, ('L', 'R'): -3, ('S', 'P'): 1, ('V', 'T'): 0, 
    ('Q', 'Q'): 4, ('N', 'A'): 0, ('Z', 'Y'): -4, ('W', 'R'): 2, 
    ('Q', 'A'): 0, ('S', 'D'): 0, ('H', 'H'): 6, ('S', 'H'): -1, 
    ('H', 'D'): 1, ('L', 'N'): -3, ('W', 'A'): -6, ('Y', 'M'): -2, 
    ('G', 'R'): -3, ('Y', 'I'): -1, ('Y', 'E'): -4, ('B', 'Y'): -3, 
    ('Y', 'A'): -3, ('V', 'D'): -2, ('B', 'S'): 0, ('Y', 'Y'): 10, 
    ('G', 'N'): 0, ('E', 'C'): -5, ('Y', 'Q'): -4, ('Z', 'Z'): 3, 
    ('V', 'A'): 0, ('C', 'C'): 12, ('M', 'R'): 0, ('V', 'E'): -2, 
    ('T', 'N'): 0, ('P', 'P'): 6, ('V', 'I'): 4, ('V', 'S'): -1, 
    ('Z', 'P'): 0, ('V', 'M'): 2, ('T', 'F'): -3, ('V', 'Q'): -2, 
    ('K', 'K'): 5, ('P', 'D'): -1, ('I', 'H'): -2, ('I', 'D'): -2, 
    ('T', 'R'): -1, ('P', 'L'): -3, ('K', 'G'): -2, ('M', 'N'): -2, 
    ('P', 'H'): 0, ('F', 'Q'): -5, ('Z', 'G'): 0, ('X', 'L'): -1, 
    ('T', 'M'): -1, ('Z', 'C'): -5, ('X', 'H'): -1, ('D', 'R'): -1, 
    ('B', 'W'): -5, ('X', 'D'): -1, ('Z', 'K'): 0, ('F', 'A'): -3, 
    ('Z', 'W'): -6, ('F', 'E'): -5, ('D', 'N'): 2, ('B', 'K'): 1, 
    ('X', 'X'): -1, ('F', 'I'): 1, ('B', 'G'): 0, ('X', 'T'): 0, 
    ('F', 'M'): 0, ('B', 'C'): -4, ('Z', 'I'): -2, ('Z', 'V'): -2, 
    ('S', 'S'): 2, ('L', 'Q'): -2, ('W', 'E'): -7, ('Q', 'R'): 1, 
    ('N', 'N'): 2, ('W', 'M'): -4, ('Q', 'C'): -5, ('W', 'I'): -5, 
    ('S', 'C'): 0, ('L', 'A'): -2, ('S', 'G'): 1, ('L', 'E'): -3, 
    ('W', 'Q'): -5, ('H', 'G'): -2, ('S', 'K'): 0, ('Q', 'N'): 1, 
    ('N', 'R'): 0, ('H', 'C'): -3, ('Y', 'N'): -2, ('G', 'Q'): -1, 
    ('Y', 'F'): 7, ('C', 'A'): -2, ('V', 'L'): 2, ('G', 'E'): 0, 
    ('G', 'A'): 1, ('K', 'R'): 3, ('E', 'D'): 3, ('Y', 'R'): -4, 
    ('M', 'Q'): -1, ('T', 'I'): 0, ('C', 'D'): -5, ('V', 'F'): -1, 
    ('T', 'A'): 1, ('T', 'P'): 0, ('B', 'P'): -1, ('T', 'E'): 0, 
    ('V', 'N'): -2, ('P', 'G'): 0, ('M', 'A'): -1, ('K', 'H'): 0, 
    ('V', 'R'): -2, ('P', 'C'): -3, ('M', 'E'): -2, ('K', 'L'): -3, 
    ('V', 'V'): 4, ('M', 'I'): 2, ('T', 'Q'): -1, ('I', 'G'): -3, 
    ('P', 'K'): -1, ('M', 'M'): 6, ('K', 'D'): 0, ('I', 'C'): -2, 
    ('Z', 'D'): 3, ('F', 'R'): -4, ('X', 'K'): -1, ('Q', 'D'): 2, 
    ('X', 'G'): -1, ('Z', 'L'): -3, ('X', 'C'): -3, ('Z', 'H'): 2, 
    ('B', 'L'): -3, ('B', 'H'): 1, ('F', 'F'): 9, ('X', 'W'): -4, 
    ('B', 'D'): 3, ('D', 'A'): 0, ('S', 'L'): -3, ('X', 'S'): 0, 
    ('F', 'N'): -3, ('S', 'R'): 0, ('W', 'D'): -7, ('V', 'Y'): -2, 
    ('W', 'L'): -2, ('H', 'R'): 2, ('W', 'H'): -3, ('H', 'N'): 2, 
    ('W', 'T'): -5, ('T', 'T'): 3, ('S', 'F'): -3, ('W', 'P'): -6, 
    ('L', 'D'): -4, ('B', 'I'): -2, ('L', 'H'): -2, ('S', 'N'): 1, 
    ('B', 'T'): 0, ('L', 'L'): 6, ('Y', 'K'): -4, ('E', 'Q'): 2, 
    ('Y', 'G'): -5, ('Z', 'S'): 0, ('Y', 'C'): 0, ('G', 'D'): 1, 
    ('B', 'V'): -2, ('E', 'A'): 0, ('Y', 'W'): 0, ('E', 'E'): 4, 
    ('Y', 'S'): -3, ('C', 'N'): -4, ('V', 'C'): -2, ('T', 'H'): -1, 
    ('P', 'R'): 0, ('V', 'G'): -1, ('T', 'L'): -2, ('V', 'K'): -2, 
    ('K', 'Q'): 1, ('R', 'A'): -2, ('I', 'R'): -2, ('T', 'D'): 0, 
    ('P', 'F'): -5, ('I', 'N'): -2, ('K', 'I'): -2, ('M', 'D'): -3, 
    ('V', 'W'): -6, ('W', 'W'): 17, ('M', 'H'): -2, ('P', 'N'): 0, 
    ('K', 'A'): -1, ('M', 'L'): 4, ('K', 'E'): 0, ('Z', 'E'): 3, 
    ('X', 'N'): 0, ('Z', 'A'): 0, ('Z', 'M'): -2, ('X', 'F'): -2, 
    ('K', 'C'): -5, ('B', 'Q'): 1, ('X', 'B'): -1, ('B', 'M'): -2, 
    ('F', 'C'): -4, ('Z', 'Q'): 3, ('X', 'Z'): -1, ('F', 'G'): -5, 
    ('B', 'E'): 3, ('X', 'V'): -1, ('F', 'K'): -5, ('B', 'A'): 0, 
    ('X', 'R'): -1, ('D', 'D'): 4, ('W', 'G'): -7, ('Z', 'F'): -5, 
    ('S', 'Q'): -1, ('W', 'C'): -8, ('W', 'K'): -3, ('H', 'Q'): 3, 
    ('L', 'C'): -6, ('W', 'N'): -4, ('S', 'A'): 1, ('L', 'G'): -4, 
    ('W', 'S'): -2, ('S', 'E'): 0, ('H', 'E'): 1, ('S', 'I'): -1, 
    ('H', 'A'): -1, ('S', 'M'): -2, ('Y', 'L'): -1, ('Y', 'H'): 0, 
    ('Y', 'D'): -4, ('E', 'R'): -1, ('X', 'P'): -1, ('G', 'G'): 5, 
    ('G', 'C'): -3, ('E', 'N'): 1, ('Y', 'T'): -3, ('Y', 'P'): -5, 
    ('T', 'K'): 0, ('A', 'A'): 2, ('P', 'Q'): 0, ('T', 'C'): -2, 
    ('V', 'H'): -2, ('T', 'G'): 0, ('I', 'Q'): -2, ('Z', 'T'): -1, 
    ('C', 'R'): -4, ('V', 'P'): -1, ('P', 'E'): -1, ('M', 'C'): -5, 
    ('K', 'N'): 1, ('I', 'I'): 5, ('P', 'A'): 1, ('M', 'G'): -3, 
    ('T', 'S'): 1, ('I', 'E'): -2, ('P', 'M'): -2, ('M', 'K'): 0, 
    ('I', 'A'): -1, ('P', 'I'): -2, ('R', 'R'): 6, ('X', 'M'): -1, 
    ('L', 'I'): 2, ('X', 'I'): -1, ('Z', 'B'): 2, ('X', 'E'): -1, 
    ('Z', 'N'): 1, ('X', 'A'): 0, ('B', 'R'): -1, ('B', 'N'): 2, 
    ('F', 'D'): -6, ('X', 'Y'): -2, ('Z', 'R'): 0, ('F', 'H'): -2, 
    ('B', 'F'): -4, ('F', 'L'): 2, ('X', 'Q'): -1, ('B', 'B'): 3
}

class GlobalAligner:
    """
    Needleman-Wunsch global alignment algorithm using highroad

    Recurrence Relation:
        F[i,j] = max(F[i-1,j-1]+s(x_i, y_i) , F[i,j-1]-d, F[i-1,j]-d)

    i_seq: array of sequence placed along rows of matrix (i-index)
    j_seq: array of sequence placed along columns of matrix (j-index)
    match: match score
    mismatch: mismatch score
    gap: linear gap penalty 
    mat: scoring matrix
    """
    def __init__(self, i_seq, j_seq, scoring_mat, gap):
        super().__init__()
        self.j_seq, self.i_seq, self.gap = self.init_seq(j_seq), self.init_seq(i_seq), gap

        self.scoring_mat = pam250 if scoring_mat == 'PAM250' else blosum62

        # list of optimal sequence tuples (there will only be one) - populated by traceback
        self.optimal_seqs = list()

        # convenience variables bc using them so often
        self.i_len = len(i_seq)
        self.j_len = len(j_seq)

        self.init_mat()
        # self.init_traceback_mat()

        self.align()
        # self.traceback()


    def init_mat(self):
        self.mat = np.zeros(shape=(self.i_len+1, self.j_len+1), dtype=np.int64)
        for i in range(self.i_len+1):
            self.mat[i,0] = i * self.gap
        for j in range(self.j_len+1):
            self.mat[0,j] = j * self.gap
    
    def init_seq(self, seq):
        return np.array(list(seq))
    
    # def init_traceback_mat(self):
    #     self.traceback_mat = np.empty(shape=(self.i_len+1, self.j_len+1), dtype=object)

    def align(self):
        for i in range(1, self.i_len+1):
            for j in range(1, self.j_len+1):
                # self.mat[i,j], self.traceback_mat[i,j] = self.score(i,j)
                self.mat[i,j] = self.score(i,j)
        self.alignment_score = self.mat[self.i_len, self.j_len]

    def score(self, i, j):
        s = self.scoring_mat[(self.i_seq[i-1], self.j_seq[j-1])] if (self.i_seq[i-1], self.j_seq[j-1]) in self.scoring_mat else self.scoring_mat[(self.j_seq[j-1], self.i_seq[i-1])]

        return max(self.mat[i-1,j-1] + s,
                    self.mat[i,j-1] + self.gap,
                    self.mat[i-1,j] + self.gap)

        # ret_max = max(self.mat[i-1,j-1] + s,
        #               self.mat[i,j-1] + self.gap,
        #               self.mat[i-1,j] + self.gap)

        # if ret_max == self.mat[i-1,j] + self.gap:
        #     ret_str = 'up'
        # elif ret_max == self.mat[i-1,j-1] + s:
        #     ret_str = 'diag'
        # else:
        #     ret_str = 'left'
        # return ret_max, ret_str

    # def traceback(self):
    #     """provides traceback for j sequence"""
    #     traceback_i = list()
    #     traceback_j = list()
    #     i, j = self.i_len, self.j_len
        
    #     # replacing traceback matrix values with suffix to make it easier to print out for website
    #     while i > 0 and j > 0:
    #         if self.traceback_mat[i,j] == 'diag':
    #             self.traceback_mat[i,j] = 'DIAG_OPTIMAL'
    #             traceback_i.append(self.i_seq[i-1])
    #             traceback_j.append(self.j_seq[j-1])
    #             i -= 1; j -= 1
    #         elif self.traceback_mat[i,j] == 'left':
    #             self.traceback_mat[i,j] = 'LEFT_OPTIMAL'
    #             traceback_i.append('_')
    #             traceback_j.append(self.j_seq[j-1])
    #             j -= 1
    #         else:
    #             self.traceback_mat[i,j] = 'UP_OPTIMAL'
    #             traceback_i.append(self.i_seq[i-1])
    #             traceback_j.append('_')
    #             i -= 1

    #     # if backpointer hasn't completed navigating through entire sequence (and to get the first letter in sequence)
    #     if j > 0:
    #         while j > 0:
    #             traceback_i.append('_')
    #             self.traceback_mat[i,j] = 'LEFT_OPTIMAL'
    #             traceback_j.append(self.j_seq[j-1])
    #             j -= 1
    #     elif i > 0:
    #         while i > 0:
    #             traceback_j.append('_')
    #             self.traceback_mat[i,j] = 'UP_OPTIMAL'
    #             traceback_i.append(self.i_seq[i-1])
    #             i -= 1

    #     # tuple of i and j optimal sequence
    #     self.optimal_seqs.append((''.join(traceback_i[::-1]), ''.join(traceback_j[::-1])))

class LocalAligner:
    """
    Smith-Waterman local alignment algorithm using highroad

    Recurrence Relation:
        F[i,j] = max(F[i-1,j-1]+s(x_i, y_i) , F[i,j-1]-d, F[i-1,j]-d, 0)

    i_seq: array of sequence placed along rows of matrix (i-index)
    j_seq: array of sequence placed along columns of matrix (j-index)
    match: match score
    mismatch: mismatch score
    gap: linear gap penalty 
    mat: scoring matrix
    """
    def __init__(self, i_seq, j_seq, scoring_mat, gap):
        super().__init__()
        self.j_seq, self.i_seq, self.gap = self.init_seq(j_seq), self.init_seq(i_seq), gap

        self.scoring_mat = pam250 if scoring_mat == 'PAM250' else blosum62

        # heap for traceback (score, (i-index, j-index))
        self.heap = []

        # list of optimal sequence tuples - populated by traceback
        self.optimal_seqs = list()

        # convenience variables bc using them so often
        self.i_len = len(i_seq)
        self.j_len = len(j_seq)

        self.init_mat()
        # self.init_traceback_mat()

        self.align()
        # self.traceback()

    def init_mat(self):
        self.mat = np.zeros(shape=(self.i_len+1, self.j_len+1), dtype=np.int)
    
    def init_seq(self, seq):
        return np.array(list(seq))
    
    # def init_traceback_mat(self):
    #     self.traceback_mat = np.empty(shape=(self.i_len+1, self.j_len+1), dtype=object)

    def align(self):
        for i in range(1, self.i_len+1):
            for j in range(1, self.j_len+1):
                # self.mat[i,j], self.traceback_mat[i,j] = self.score(i,j)
                self.mat[i,j] = self.score(i,j)
        self.alignment_score = -self.heap[0][0]

    def score(self, i, j):
        s = self.scoring_mat[(self.i_seq[i-1], self.j_seq[j-1])] if (self.i_seq[i-1], self.j_seq[j-1]) in self.scoring_mat else self.scoring_mat[(self.j_seq[j-1], self.i_seq[i-1])]

        ret_max = max(self.mat[i-1,j-1] + s,
                      self.mat[i,j-1] + self.gap,
                      self.mat[i-1,j] + self.gap,
                      0)

        # if ret_max == self.mat[i-1,j] + self.gap:
        #     ret_str = 'up'
        # elif ret_max == self.mat[i-1,j-1] + s:
        #     ret_str = 'diag'
        # elif ret_max == self.mat[i,j-1] + self.gap:
        #     ret_str = 'left'
        # else:
        #     ret_str = 'zero'

        # need to push negative max onto heap to mimic max heap
        hp.heappush(self.heap, (-ret_max, (i,j)))

        # return ret_max, ret_str
        return ret_max

    # def traceback(self):
    #     """provides traceback for j sequence"""
    #     # loops until a cell with less max score is found - can be ties in local alignment
    #     max = -self.heap[0][0]
    #     while (-self.heap[0][0] == max):
    #         traceback_i = list()
    #         traceback_j = list()

    #         cell_score, cell = hp.heappop(self.heap)
    #         cell_score *= -1
    #         i, j = cell[0], cell[1]
            
    #         # need to loop until hit a 0
    #         while self.mat[i,j] > 0:
    #             if self.traceback_mat[i,j] == 'diag':
    #                 self.traceback_mat[i,j] = 'DIAG_OPTIMAL'
    #                 traceback_i.append(self.i_seq[i-1])
    #                 traceback_j.append(self.j_seq[j-1])
    #                 i -= 1; j -= 1
    #             elif self.traceback_mat[i,j] == 'left':
    #                 self.traceback_mat[i,j] = 'LEFT_OPTIMAL'
    #                 traceback_i.append('_')
    #                 traceback_j.append(self.j_seq[j-1])
    #                 j -= 1
    #             elif self.traceback_mat[i,j] == 'up':
    #                 self.traceback_mat[i,j] = 'UP_OPTIMAL'
    #                 traceback_i.append(self.i_seq[i-1])
    #                 traceback_j.append('_')
    #                 i -= 1
    #             else:
    #                 break

    #         self.optimal_seqs.append((''.join(traceback_i[::-1]), ''.join(traceback_j[::-1])))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--infile', type=str, required=True)
    parser.add_argument('--scope', type=str, choices = ['global', 'local'], required=True)
    parser.add_argument('--gap', type=int, required=True)
    parser.add_argument('--db', type=str, required=True)
    parser.add_argument('--matrix', type=str, choices = ['PAM250', 'BLOSUM62'], required=True)
    parser.add_argument('--outfile', type=str, default='align.txt')
    
    args = parser.parse_args()

    
    with open(args.outfile, "w") as f:
        for record in SeqIO.parse(args.infile, "fasta"):
            header, seq = record.description, record.seq
        db = args.db
        scope = args.scope
        gap = args.gap

        f.write(f'Query Sequence:\n>{seq}\n\nDatabase: {db}\n\n')

        # https://stackoverflow.com/questions/5622976/how-do-you-calculate-program-run-time-in-python
        start = timeit.default_timer()

        if scope == 'global':
            scores = [f'> {GlobalAligner(seq, str(record.seq), args.matrix, -gap).alignment_score} | {record.description}\n' for record in SeqIO.parse(db, "fasta")]
        else:
            scores = [f'> {LocalAligner(seq, str(record.seq), args.matrix, -gap).alignment_score} | {record.description}\n' for record in SeqIO.parse(db, "fasta")]

        stop = timeit.default_timer()
        runtime = stop - start

        f.write(f'Implementation: Serial\nRuntime: {runtime}\n\n')

        for line in scores:
            f.write(line)

    