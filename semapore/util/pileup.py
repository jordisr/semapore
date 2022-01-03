import re
import pysam
import numpy as np
import pandas as pd
from Bio import SeqIO

def load_fastx(f, fmt="fasta"):
    # read FASTA/FASTQ into memory
    seqs = []
    with open(f) as handle:
        for record in SeqIO.parse(handle, fmt):
            seqs.append((record.id, str(record.seq)))
    return seqs

def get_pileup_alignment(alignment, reference):
    """
    Generate pileup alignment table of reads aligned to the reference / draft assembly.

    Takes alignment file in BAM and reference in FASTA format.
    """

    (ref_id, ref_seq) = load_fastx(reference, "fasta")[0]

    del_marker = "<DEL>"
    ins_marker = "<INS>"
    no_marker = "<NONE>"

    alignment_columns = []

    with pysam.AlignmentFile(alignment, "rb") as samfile:

        for pileupcolumn in samfile.pileup(min_base_quality=0):

            #print(pileupcolumn.pos, pileupcolumn.n)
            query_sequences = pileupcolumn.get_query_sequences(add_indels=True, mark_matches=True, mark_ends=False)
            query_names = pileupcolumn.get_query_names()
            query_positions = pileupcolumn.get_query_positions()
            this_column = {ref_id:(pileupcolumn.pos, ref_seq[pileupcolumn.pos])}
            insertions = {}
            max_insert_size = 0

            for base,read,pos in zip(query_sequences, query_names, query_positions):
                if len(base) > 1:
                    rem = re.match(r"([ACGTacgt])([-+])(\d+)([AaCcGgTtNn]+)",base)
                    base_before, indel_type, indel_size, indel_seq = rem.groups(0)
                    this_column[read] = (pos,base_before)

                    # keep track of insertions to add columns to alignment later
                    if indel_type == '+':
                        insertions[read] = (pos,indel_seq)
                        if int(indel_size) > max_insert_size:
                            max_insert_size = int(indel_size)

                elif base == '*':
                    this_column[read] = del_marker
                else:
                    this_column[read] = (pos, base)

            alignment_columns.append(this_column)

            # add insertion columns
            for i in range(max_insert_size):
                insertion_col = {ref_id:ins_marker}
                for k,v in insertions.items():
                    if i < len(v[1]):
                        insertion_col[k] = (v[0]+1+i, v[1][i])
                    else:
                        insertion_col[k] = ins_marker
                for r in query_names:
                    if r not in insertions:
                        insertion_col[r] = ins_marker
                alignment_columns.append(insertion_col)

    return pd.DataFrame(alignment_columns).fillna(no_marker)
