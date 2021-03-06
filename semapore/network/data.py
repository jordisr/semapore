import os
import sys
import copy
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import pandas as pd
import mappy

import semapore.util

def get_alignment_coord(cs):
    """Parse CS string of mappy.Alignment
    """
    operation = ""
    operation_field = ""

    q2r = []
    r2q = []
    coord_r = -1
    coord_q = -1

    for i, c in enumerate(cs):
        if c in [':','+','-','*',"~"] or (i == len(cs)-1):
            if (i == len(cs)-1):
                operation_field += c
            if operation != "":
                # at the start of next field do something for previous
                if operation == ":":
                    match_length = int(operation_field)

                    for j in range(match_length):
                        coord_r += 1
                        coord_q += 1
                        r2q.append(coord_q)
                        q2r.append(coord_r)

                elif operation == "+":
                    # insertion with respect to the reference
                    for j in range(len(operation_field)):
                        q2r.append(coord_r)
                    coord_q += len(operation_field)

                elif operation == "-":
                    # deletion with respect to the reference
                    for j in range(len(operation_field)):
                        r2q.append(coord_q)
                    coord_r += len(operation_field)

                elif operation == "*":
                    # substitution
                    coord_r += 1
                    coord_q += 1
                    r2q.append(coord_q)
                    q2r.append(coord_r)

                elif operation == "~":
                    pass
            operation = c
            operation_field = ""
        else:
            operation_field += c

    r2q = np.array(r2q)
    q2r = np.array(q2r)

    r2q[r2q < 0] = 0
    q2r[q2r < 0] = 0

    return [r2q, q2r]

def add_labels_from_reference(features, aligner, mappy_batch_size=None):
    features_with_labels = []
    dna_alphabet = np.array(['A','C','G','T'])
    
    if mappy_batch_size is None:
        mappy_batch_size = len(features)
        
    for mappy_batch_start in range(0, len(features), mappy_batch_size):

        features_batch = features[mappy_batch_start:mappy_batch_start+mappy_batch_size]

        draft_window_seq = [[(x - 3) % 4 for x in y['draft'] if x > 2] for y in features_batch]
        draft_window_len = np.array([len(x) for x in draft_window_seq])
        draft_window_pos = np.cumsum(draft_window_len)
        draft_seq = np.concatenate(draft_window_seq).astype(np.int16)
        draft_seq_to_map = ''.join(np.take(dna_alphabet, draft_seq))

        draft_seq_bounds = np.zeros((draft_window_pos.shape[0], 2)).astype(int)
        draft_seq_bounds[1:,0] = draft_window_pos[:-1]
        draft_seq_bounds[:,1] = draft_window_pos

        try:
            hit = semapore.network.AlignmentCopy(aligner, next(aligner.map(draft_seq_to_map, cs=True)))
        except:
            print("No alignment", file=sys.stderr)
            continue
        #print(hit.mlen/hit.blen, file=sys.stderr)

        r_seq = hit.r_seq

        #print("strand:{},  q_st:{}, q_en:{}".format(hit.strand, hit.q_st, hit.q_en))
        [r2q, q2r] = semapore.network.get_alignment_coord(hit.cs)

        # reverse coordinates for reverse matches
        if hit.strand < 0:
            r_seq = semapore.util.revcomp(r_seq)
            #r2q = max(r2q)-r2q[::-1]
            q2r = max(q2r)-q2r[::-1]

        #print(q2r, len(r_seq), len(draft_seq), len(draft_seq_to_map), hit.q_en - hit.q_st, len(q2r))

        for (draft_start, draft_end), feature in zip(draft_seq_bounds, features_batch):
            if draft_start >= hit.q_st and draft_end <= hit.q_en and (draft_end-draft_start) > 0:
                ref_seq_from_window = r_seq[q2r[draft_start-hit.q_st]:q2r[draft_end-hit.q_st-1]+1]
                if 'N'in ref_seq_from_window or len(ref_seq_from_window) < 1 or (draft_end - draft_start) < 1 and len(feature['signal_values']) > 0:
                    print("bad reference", len(ref_seq_from_window), draft_end-draft_start, file=sys.stderr)
                    continue
                
                ref_seq_from_window = np.array([{'A':0,'C':1,'G':2,'T':3}[i] for i in ref_seq_from_window])
                ref_seq_from_window = ref_seq_from_window.astype(np.int16)

                this_feature = copy.deepcopy(feature)
                this_feature['labels'] = ref_seq_from_window
                features_with_labels.append(this_feature)
            else:
                print("not fully contained", (draft_start, draft_end), (hit.q_st, hit.q_en), file=sys.stderr)
                continue                

    return features_with_labels

def get_reference_sequences(pileup, window_bounds, hit):
    """Get true labels by aligning draft to reference

    Args:
        pileup (Pileup)
        window_bounds (ndarray): 2D array from featurize_inputs()
        aligner (mappy.Aligner): aligner to reference genome

    Returns:
        ndarray: draft_window_sequences, draft sequence strings
        ndarray: ref_window_sequences, ref sequence strings
        ndarray: ref_window_idx, positions with valid ref seq output
        str: ref_name, reference sequence that draft aligend to
    """

    # extract draft sequence from parsed pileup
    draft_pileup = pileup.pileup[pileup.ref_id]
    draft_sequence = ''.join([['','','','A','C','G','T','A','C','G','T'][x[1]] for x in draft_pileup])

    # get first hit from alignment(s) to reference
    #hit = next(aligner.map(draft_sequence, cs=True))
    ref_name = hit.ctg

    # get pairwise alignment coordinates
    r_seq = hit.r_seq
    [r2q, q2r] = get_alignment_coord(hit.cs)
    if hit.strand < 0:
        # reverse coordinates for reverse matches
        r_seq = semapore.util.revcomp(r_seq)
        r2q = max(r2q)-r2q[::-1]
        q2r = max(q2r)-q2r[::-1]

    #print("Identity:{} Strand:{}".format(hit.mlen/hit.blen, hit.strand), file=sys.stderr)

    draft_window_sequences = []
    ref_window_sequences = []
    ref_window_idx = []

    window_size = window_bounds[0,1]-window_bounds[0,0]

    for window_idx, pileup_window in enumerate(window_bounds):
        draft_pileup_window = list(draft_pileup.iloc[pileup_window[0]:pileup_window[1]])

        try:
            # TODO: be explicit about behavior here
            # go top down, find first non insert character
            i = 0
            while draft_pileup_window[i][1] < 3:
                i += 1
            draft_start = draft_pileup_window[i][0]

            # go bottom up, find first non insert character
            i = len(draft_pileup_window)-1
            while draft_pileup_window[i][1] < 3:
                i -= 1
            draft_end = draft_pileup_window[i][0]
        except IndexError:
            # skip sections with all inserts for draft sequence
            print("Warning: all inserts for draft, window {}, alignment={}".format(window_idx, pileup.alignment),  file=sys.stderr)
            continue

        # check this chunk of draft contained in alignment to ref
        if draft_start >= hit.q_st and draft_end <= hit.q_en:
            ref_rel_start = q2r[draft_start-hit.q_st]
            ref_rel_end = q2r[draft_end-hit.q_st]

            # add one for inclusive ranges
            this_ref_seq = r_seq[ref_rel_start:ref_rel_end + 1]
            this_draft_seq = draft_sequence[draft_start:draft_end + 1]

            if 'N' in this_ref_seq:
                print("Warning: N in ref seq, window {}, draft_len={} ref_len={}, alignment={}".format(window_idx, len(this_draft_seq), len(this_ref_seq), pileup.alignment),  file=sys.stderr)            
            elif len(this_ref_seq) > 0 and len(this_draft_seq) > 0 and len(this_ref_seq) <= window_size:
                ref_window_idx.append(window_idx)
                ref_window_sequences.append(np.array([{'A':0,'C':1,'G':2,'T':3}[i] for i in this_ref_seq]).astype(np.int16))
                draft_window_sequences.append(np.array([{'A':0,'C':1,'G':2,'T':3}[i] for i in this_draft_seq]).astype(np.int16))
                #print("{}\t{}..{}\t{}..{}".format(hit.strand, this_draft_seq[:10],this_draft_seq[-10:], this_ref_seq[:10],this_ref_seq[-10:]), file=sys.stderr)

            else:
                print("Warning: bad chunk, window {}, draft_len={} ref_len={}, alignment={}".format(window_idx, len(this_draft_seq), len(this_ref_seq), pileup.alignment),  file=sys.stderr)
        else:
            print("Warning: draft not contained in ref, window {}, alignment={}".format(window_idx, pileup.alignment),  file=sys.stderr)

    return draft_window_sequences, ref_window_sequences, np.array(ref_window_idx), np.array([ref_name, hit.strand, hit.mlen/hit.blen, hit.q_st, hit.q_en, hit.r_st, hit.r_en])

def make_training_data(draft, alignment, reads, hit=None, out="tmp", reference="", trimmed_reads="", window=64, max_time=80):
    """Write labeled training data for one draft sequence to NPZ

    Args:
        draft (str): path to draft assembly sequence in FASTA
        alignment (str): path to BAM alignment
        reads (str/dict): path to directory or dict {read_id:path}
        out (str): name to save file under
        aligner (mappy.Aligner): aligner to reference genome
        reference (str): path to reference genome FASTA (needed if aligner=None)
        trimmed_reads (str): path to trimmed reads (FASTA)
        window (int): number of pileup alignment columns per input
        max_time (int): max signal/event size, larger ones will be truncated
    """

    # load alignment
    pileup = semapore.util.Pileup(alignment=alignment, reference=draft)

    # load reads
    if type(reads) is str:
        reads = semapore.util.get_reads(pileup.reads, dir=reads)
    elif type(reads) is dict:
        reads = semapore.util.get_reads(pileup.reads, paths=reads)
    else:
        sys.exit("No reads provided!")

    # load trimmed read sequences
    if trimmed_reads:
        semapore.util.trim_raw_reads(reads, trimmed_reads)

    # batch alignment and reads for inference
    sequence_input, signal_input, signal_input_mask, window_bounds = semapore.util.featurize_inputs(pileup, reads, window_size=window, max_time=max_time, draft_first=True)
    draft_input = sequence_input[:,:,0] 
    sequence_input = sequence_input[:,:,1:]

    #print("Moving to draft: {}".format(draft), file=sys.stderr)
    draft_sequences, ref_sequences, ref_windows, ref_name = get_reference_sequences(pileup, window_bounds, hit)

    print("{}: {} out of {} windows for training data".format(alignment, len(ref_windows), len(window_bounds)))
    files = np.array([draft, alignment, reference, trimmed_reads])
    draft_input = draft_input[ref_windows]
    sequence_input = sequence_input[ref_windows]
    signal_input = signal_input[ref_windows]
    signal_input_mask = signal_input_mask[ref_windows]
    
    outfile = "{}.all.npz".format(out)
    np.savez_compressed(outfile,
        files=files,
        draft_input=draft_input,
        sequence_input=sequence_input,
        signal_input=signal_input,
        signal_mask=signal_input_mask,
        ref_sequences=ref_sequences,
        draft_sequences=draft_sequences,
        ref_name=ref_name)

    # save windows where draft and reference aren't identical
    ref_is_diff =  ~(ref_sequences == draft_sequences)
    #print(np.sum(ref_is_diff), file=sys.stderr)
    #print(ref_sequences[0])
    #print(draft_sequences[0])
    if ref_is_diff.any():
        outfile = "{}.errors.npz".format(out)
        np.savez_compressed(outfile,
            files=files,
            draft_input=draft_input[ref_is_diff],
            sequence_input=sequence_input[ref_is_diff],
            signal_input=signal_input[ref_is_diff],
            signal_mask=signal_input_mask[ref_is_diff],
            ref_sequences=ref_sequences[ref_is_diff],
            draft_sequences=draft_sequences[ref_is_diff],
            ref_name=ref_name)

        outfile = "{}.same.npz".format(out)
        np.savez_compressed(outfile,
            files=files,
            draft_input=draft_input[~ref_is_diff],
            sequence_input=sequence_input[~ref_is_diff],
            signal_input=signal_input[~ref_is_diff],
            signal_mask=signal_input_mask[~ref_is_diff],
            ref_sequences=ref_sequences[~ref_is_diff],
            draft_sequences=draft_sequences[~ref_is_diff],
            ref_name=ref_name)

    row = [os.path.dirname(draft).split('/')[-1], np.sum(ref_is_diff), len(draft_input)-np.sum(ref_is_diff)] + list(ref_name)
    return row

class TrainingDataHelper:
    def __init__(self, draft, alignment, reads, reference, trimmed_reads="", out="training", window=64, max_time=80):
        self.out = out
        self.draft = draft
        self.alignment = alignment
        self.reads = reads
        self.trimmed_reads = trimmed_reads
        self.window = window
        self.max_time = max_time
        self.reference = reference

    def __call__(self, x):
        f, hit = x
        try:
            return make_training_data(
                out="{}/{}".format(self.out, os.path.basename(f)),
                draft="{}/{}".format(f, self.draft),
                alignment="{}/{}".format(f, self.alignment),
                trimmed_reads="{}/{}".format(f, self.trimmed_reads),
                reads=self.reads,
                hit=hit,
                window=self.window,
                max_time=self.max_time)
        except:
            print("WARNING: Caught error with {}".format(os.path.basename(f)))
            return [os.path.basename(f)] + [None] * 9

class AlignmentCopy:
    # pickleable version of mappy.Alignment
    def __init__(self, aligner, hit):
        self.ctg = hit.ctg
        self.strand = hit.strand
        self.cs = hit.cs
        self.q_st = hit.q_st
        self.q_en = hit.q_en
        self.r_st = hit.r_st
        self.r_en = hit.r_en
        self.mlen = hit.mlen
        self.blen = hit.blen
        self.r_seq = aligner.seq(hit.ctg, start=hit.r_st, end=hit.r_en)

def make_training_dir(dirs, draft, alignment, reads, reference, trimmed_reads="", out="training", window=64, max_time=80, threads=1):
    """Make training NPZ files for nested directory of assemblies

    Args:
        dirs (list): list of directories
        alignment (str): path to BAM alignment
        draft (str): path to draft (FASTA)
        reads (str/dict): directory or dict of read paths
        reference (str): path to reference genome FASTA
        trimmed_reads (str): path to trimmed reads (FASTA)
        out (str): name of output directory to create
        window (int): number of pileup alignment columns per input
        max_time (int): max signal/event size, larger ones will be truncated
    """

    os.makedirs(out)

    if threads > 1:
        aligner = mappy.Aligner(reference, preset='map-ont', n_threads=1)
        training_input = []
        print("aligning to reference...", file=sys.stderr)
        for f in tqdm(dirs):
            draft_path = "{}/{}".format(f, draft)
            seqs = semapore.util.load_fastx(draft_path, "fasta")
            if len(seqs) > 0 and len(seqs[0][1]) > 0 :
                hit = AlignmentCopy(aligner, next(aligner.map(seqs[0][1], cs=True)))
                training_input.append((f, hit))

        training_helper = TrainingDataHelper(
                            out=out,
                            reference=reference,
                            draft=draft,
                            alignment=alignment,
                            trimmed_reads=trimmed_reads,
                            reads=reads,
                            window=window,
                            max_time=max_time)
        print("starting multiprocessing...", file=sys.stderr)
        with Pool(processes=threads) as pool:
            #map_output = pool.map(training_helper, training_input, chunksize=20)
            map_output = list(tqdm(pool.imap(training_helper, training_input, chunksize=20), total=len(training_input)))
            #map_output = pool.imap(training_helper, training_input, chunksize=20)
        map_output = pd.DataFrame(map_output, columns=['name','num_diff','num_same','genome','strand','identity','query_start','query_end','ref_start','ref_end'])
        map_output.to_csv(os.path.join(out,"log.csv"), index=False)
    else:
        aligner = mappy.Aligner(reference, preset='map-ont')

        for f in dirs:
            try:
                make_training_data(
                    out="{}/{}".format(out, os.path.basename(f)),
                    draft="{}/{}".format(f, draft),
                    alignment="{}/{}".format(f, alignment),
                    trimmed_reads="{}/{}".format(f, trimmed_reads),
                    reads=reads,
                    aligner=aligner,
                    window=window,
                    max_time=max_time)
            except:
                pass
