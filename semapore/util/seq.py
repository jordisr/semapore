from Bio import SeqIO

def revcomp(s):
    return ''.join([{'A':'T','C':'G','G':'C','T':'A'}[x] for x in s[::-1]])

def load_fastx(f, fmt="fasta"):
    # read FASTA/FASTQ into memory
    seqs = []
    with open(f) as handle:
        for record in SeqIO.parse(handle, fmt):
            seqs.append((record.id, str(record.seq)))
    return seqs

def fasta_format(name, seq, width=80):
    fasta = '>'+name+'\n'
    window = 0
    while window+width < len(seq):
        fasta += (seq[window:window+width]+'\n')
        window += width
    fasta += (seq[window:]+'\n')
    return(fasta)

def get_sequence_offset(full_seq, sub_seq):
    len_full = len(full_seq)
    len_sub = len(sub_seq)
    sub_start =  full_seq.find(sub_seq)
    rel_start = 0
    rel_end = 0
    if sub_start >= 0:
        rel_start = sub_start
        rel_end = sub_start+len_sub
    if (rel_end > 0) and (rel_end <= len_full):
        return rel_start, rel_end
    else:
        return None

def trim_raw_reads(reads, path_to_trimmed):
    """Apply sequence offsets to FAST5 segmentation/sequence data (in place)

    Args:
        reads (dict): dict of dicts returned by get_reads()
        path_to_trimmed (str): path to FASTA file of trimmed reads
    """
    trimmed_reads = load_fastx(path_to_trimmed, "fasta")
    read_offsets = {}
    for r in trimmed_reads:
        subread = r[0].split('_')
        if r[0] in reads:
            full_seq = reads[r[0]]['sequence']
            sub_seq = r[1]
            read_offset = get_sequence_offset(full_seq=full_seq, sub_seq=sub_seq)
            read_offsets[r[0]] = read_offset
            if read_offset:
                #print(full_seq[read_offset[0]:read_offset[0]+10],sub_seq[:10])
                reads[r[0]]['segments'] = reads[r[0]]['segments'][read_offset[0]:read_offset[1]]
                reads[r[0]]['sequence'] = reads[r[0]]['sequence'][read_offset[0]:read_offset[1]]
                assert(reads[r[0]]['sequence'] == sub_seq)
            else:
                print("Failed to calculate read offset!")
