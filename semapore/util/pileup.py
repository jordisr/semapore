import re
import sys
import pysam
import numpy as np
import pandas as pd
from tqdm import tqdm
from Bio import SeqIO

def load_fastx_dict(f, fmt="fasta"):
    # read FASTA/FASTQ into memory
    seqs = {}
    with open(f) as handle:
        for record in SeqIO.parse(handle, fmt):
            seqs[record.id] = str(record.seq)
    return seqs

class ReadID:
    def __init__(self):
        self.n = -1
        self.dict = {}
        self.reads = []

    def __getitem__(self, key):
        if key in self.dict:
            return self.dict[key]
        else:
            self.n += 1
            self.dict[key] = self.n
            self.reads.append(key)
            return self.n

    def __len__(self):
        return len(self.dict)

class Pileup:
    def __init__(self, alignment_path, reference_path, refname):
        self.alignment_path = alignment_path
        self.reference_path = reference_path
        self.refname = refname
        self.refseq = []
        self.pileup = []
        self.read_id = ReadID()

    def add_columm(self, column):
        self.pileup.append({})
        for k,v in column.items():
            if k == self.refname:
                self.refseq.append(v)
            else:
                self.pileup[-1][self.read_id[k]] = v

    def get_num_reads(self):
        return self.read_id.n + 1

    def get_num_columns(self):
        return len(self.pileup)

    def __len__(self):
        return self.get_num_columns()

    def get_window(self, start, end, min_overlap=None):
        # return pileup segment with all reads with minimum overlap over window
        window = pd.DataFrame(self.pileup[start:end]).applymap(lambda x: (None,0) if pd.isnull(x) else x)
        if min_overlap is not None:
            cols_to_drop = list(window.columns[np.sum(window.applymap(lambda x: x[1] == 0)) > int(min_overlap*(end-start))])
            window.drop(columns=cols_to_drop, inplace=True)
        window_reads = [self.read_id.reads[x] for x in window.columns]
        window_pos = window.applymap(lambda x: x[0]).to_numpy().astype(int)
        window_pileup = window.applymap(lambda x: x[1]).to_numpy().astype(int)
        window_refseq = np.array(self.refseq[start:end])[:,1]
        return PileupWindow(reads=window_reads, refseq=window_refseq, pos=window_pos, pileup=window_pileup)

    def get_reads(self):
        return list(self.read_id.dict.keys())

    def get_depth(self):
        return np.array([len(x) for x in self.pileup])

class PileupWindow:
    def __init__(self, reads, refseq, pileup, pos):
        self.reads = reads
        self.pos = pos
        self.pileup = pileup
        self.refseq = refseq

def get_pileup(alignment, reference):
    """Generate pileup alignment of reads aligned to the draft assembly/reference

    Args:
        alignment (str): path to BAM alignment
        reference (str): path to alignment reference in FASTA format

    Returns:
        List of Pileup objects
    """

    ref_seqs = load_fastx_dict(reference, "fasta")

    char_encoding = {v:i for i,v in enumerate(['<NONE>','<DEL>','<INS>','A','C','G','T','a','c','g','t'])}
    null_value = np.iinfo(np.int32).max
    
    sequence_pileups = []
    pysam.set_verbosity(0)
    
    with pysam.AlignmentFile(alignment, "rb") as samfile:
        with tqdm(total=len(ref_seqs)) as pbar:
            pileupref = None
            for pileupcolumn in samfile.pileup(min_base_quality=0):
                if pileupcolumn.reference_name != pileupref:
                    if pileupref:
                        sequence_pileups.append(curr_pileup)
                    pileupref = pileupcolumn.reference_name                    
                    curr_pileup = Pileup(alignment_path=alignment, reference_path=reference, refname=pileupref)
                    pbar.update(1)

                query_sequences = pileupcolumn.get_query_sequences(add_indels=True, mark_matches=True, mark_ends=False)
                query_names = pileupcolumn.get_query_names()
                query_positions = pileupcolumn.get_query_positions()

                this_column = {pileupref:(pileupcolumn.pos, char_encoding[ref_seqs[pileupref][pileupcolumn.pos]])}
                insertions = {}
                max_insert_size = 0

                for base,read,pos in zip(query_sequences, query_names, query_positions):
                    if len(base) > 1:
                        re_match = re.match(r"([ACGTacgt])([-+])(\d+)([AaCcGgTtNn]+)",base)
                        base_before, indel_type, indel_size, indel_seq = re_match.groups(0)
                        this_column[read] = (pos,char_encoding[base_before])

                        # keep track of insertions to add columns to alignment later
                        if indel_type == '+':
                            insertions[read] = (pos,indel_seq)
                            if int(indel_size) > max_insert_size:
                                max_insert_size = int(indel_size)

                    elif base == '*':
                        this_column[read] = (null_value, char_encoding['<DEL>'])

                    else:
                        this_column[read] = (pos, char_encoding[base])

                curr_pileup.add_columm(this_column)

                # add insertion columns
                for i in range(max_insert_size):
                    insertion_col = {pileupref:(null_value, char_encoding['<INS>'])}
                    for k,v in insertions.items():
                        if i < len(v[1]):
                            insertion_col[k] = (v[0]+1+i, char_encoding[v[1][i]])
                        else:
                            insertion_col[k] = (null_value, char_encoding['<INS>'])
                    for r in query_names:
                        if r not in insertions:
                            insertion_col[r] = (null_value, char_encoding['<INS>'])
                    curr_pileup.add_columm(insertion_col)

        sequence_pileups.append(curr_pileup)

    return sequence_pileups