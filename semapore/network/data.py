import os
import glob
import numpy as np
import mappy as mp
import tensorflow as tf

import semapore.util

def get_alignment_coord(hit):
    """Parse CS string of mappy.Alignment
    """
    cs = hit.cs
    operation = ""
    operation_field = ""

    q2r = []
    r2q = []
    coord_r = 0
    coord_q = 0

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
                    r2q.append(coord_q)
                    q2r.append(coord_r)
                    coord_r += 1
                    coord_q += 1

                elif operation == "~":
                    pass
            operation = c
            operation_field = ""
        else:
            operation_field += c

    return [np.array(r2q), np.array(q2r)]

def get_reference_sequences(draft_pileup, window_bounds, aligner):
    """Get true labels by aligning draft to reference

    Args:
        draft_pileup (Series): draft column from pileup DataFrame
        window_bounds (ndarray): 2D array from featurize_inputs()
        aligner (mappy.Aligner): aligner to reference genome

    Returns:
        ndarray: ref_window_sequences, ref sequence strings
        ndarray: ref_window_idx, positions with valid ref seq output
        ndarray: ref_window_same, bool array of whether draft matches ref
    """

    # extract draft sequence from parsed pileup
    draft_sequence = ''.join([x[1] for x in draft_pileup if type(x) == tuple])

    # get first hit from alignment(s) to reference
    hit = next(aligner.map(draft_sequence, cs=True))

    # get pairwise alignment coordinates
    r_seq = aligner.seq(hit.ctg, start=hit.r_st, end=hit.r_en)
    [r2q, q2r] = get_alignment_coord(hit)

    ref_window_sequences = []
    ref_window_idx = []
    ref_window_same = []

    for window_idx, pileup_window in enumerate(window_bounds):
        draft_pileup_window = list(draft_pileup.loc[pileup_window[0]:pileup_window[1]])

        # TODO: be explicit about behavior here
        # go top down, find first non insert character
        i = 0
        while type(draft_pileup_window[i]) is not tuple:
            i += 1
        draft_start = draft_pileup_window[i][0]

        # go bottom up, find first non insert character
        i = len(draft_pileup_window)-1
        while type(draft_pileup_window[i]) is not tuple:
            i -= 1
        draft_end = draft_pileup_window[i][0]

        # check this chunk of draft contained in alignment to ref
        if draft_start >= hit.q_st and draft_end <= hit.q_en:
            ref_rel_start = q2r[draft_start-hit.q_st]-1
            ref_rel_end = q2r[draft_end-hit.q_st]-1

            ref_window_idx.append(window_idx)
            ref_window_sequences.append(r_seq[ref_rel_start:ref_rel_end])
            ref_window_same.append(draft_sequence[draft_start:draft_end] == r_seq[ref_rel_start:ref_rel_end])

    return np.array(ref_window_sequences), np.array(ref_window_idx), np.array(ref_window_same)

def make_training_data(draft, alignment, reads_dir, out="tmp", aligner=None, reference="", trimmed_reads="", window=100, max_time=80):
    """Write labeled training data for one draft sequence to NPZ

    Args:
        draft (str): path to draft assembly sequence in FASTA
        alignment (str): path to BAM alignment
        reads_dir (str): path to directory of basecalled FAST5 reads
        out (str): name to save file under
        aligner (mappy.Aligner): aligner to reference genome
        reference (str): path to reference genome FASTA (needed if aligner=None)
        trimmed_reads (str): path to trimmed reads (FASTA)
        window (int): number of pileup alignment columns per input
        max_time (int): max signal/event size, larger ones will be truncated
    """

    # load alignment
    pileup = semapore.util.get_pileup_alignment(alignment=alignment, reference=draft)
    draft_id = pileup.columns[0]
    draft_pileup = pileup[draft_id]
    pileup.drop(draft_id, axis=1, inplace=True) # drop draft sequence row
    pileup_reads = list(pileup.columns)

    # load reads
    reads = semapore.util.get_reads(pileup_reads, base_dir=reads_dir)

    # load trimmed read sequences
    if trimmed_reads:
        semapore.util.trim_raw_reads(reads, trimmed_reads)

    # batch alignment and reads for inference
    sequence_input, signal_input, signal_input_mask, window_bounds = semapore.util.featurize_inputs(pileup, reads, window_size=window, max_time=max_time)

    # initialize mappy Aligner
    if reference and not aligner:
        aligner = mp.Aligner(reference, preset='map-ont')

    ref_seqs, ref_windows, ref_same = get_reference_sequences(draft_pileup, window_bounds, aligner)

    sequence_input = sequence_input[ref_windows]
    signal_input = signal_input[ref_windows]
    signal_input_mask = signal_input_mask[ref_windows]
    files = np.array([draft, alignment, reads_dir, reference, trimmed_reads])

    outfile = "{}.npz".format(out)
    np.savez_compressed(outfile,
        files=files,
        sequence_input=sequence_input,
        signal_input=signal_input,
        signal_mask=signal_input_mask,
        ref_sequence=ref_seqs,
        ref_same=ref_same)

def make_training_dir(pattern, draft, alignment, reads, reference, trimmed_reads="", out="training", window=64, max_time=80):
    """Make training NPZ files for nested directory of assemblies

    Args:
        pattern (str): wildcard pattern to be expanded for directory names
        alignment (str): path to BAM alignment
        draft (str): path to draft (FASTA)
        reads (str): path to directory of basecalled FAST5 reads
        reference (str): path to reference genome FASTA
        trimmed_reads (str): path to trimmed reads (FASTA)
        out (str): name of output directory to create
        window (int): number of pileup alignment columns per input
        max_time (int): max signal/event size, larger ones will be truncated
    """

    os.makedirs(out)
    aligner = mp.Aligner(reference, preset='map-ont')

    for f in glob.glob(pattern):
        make_training_data(
            out="{}/{}".format(out, os.path.basename(f)),
            draft="{}/{}".format(f, draft),
            alignment="{}/{}".format(f, alignment),
            trimmed_reads="{}/{}".format(f, trimmed_reads),
            reads_dir=reads,
            aligner=aligner,
            window=window,
            max_time=max_time)

def ragged_from_list_of_lists(l):
    return tf.RaggedTensor.from_row_lengths(np.concatenate(l), np.array([len(i) for i in l]))

def load_npz_data(f):
    training_data = np.load(f)
    x = [training_data['signal_input'], training_data['sequence_input'].astype(np.int32)]
    labels = ragged_from_list_of_lists([[{'A':0,'C':1,'G':2,'T':3}[i] for i in j] for j in training_data['ref_sequence']])
    return x, labels

def prepare_dataset(x, labels):
    x_ds = tuple([tf.data.Dataset.from_tensor_slices(x_) for x_ in x])

    labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(
        ragged_from_list_of_lists([[{'A':0,'C':1,'G':2,'T':3}[i] for i in j] for j in labels]), np.int32))

    return tf.data.Dataset.zip((x_ds, labels_ds))

def dataset_from_files(files):
    """Merge and pad NPZ data and convert to tf.data.Dataset

    Args:
        files (list): *.npz files created with make_training_dir()/make_training_data()

    Returns:
        tf.data.Dataset: dataset of zipped inputs and labels
    """

    training_data = np.load(files[0])
    (chunk_count, window_size, max_row, max_time) = training_data['signal_input'].shape
    x = [training_data['signal_input'], training_data['sequence_input'].astype(np.int32)]
    labels = training_data['ref_sequence']

    all_max_row = max_row
    all_chunk_count = chunk_count

    for f in files[1:]:
        training_data = np.load(f)
        this_x = [training_data['signal_input'], training_data['sequence_input']]
        (chunk_count, window_size, max_row, max_time) = this_x[0].shape

        if max_row > all_max_row:
            all_max_row = max_row
            x[0].resize((all_chunk_count + chunk_count, window_size, all_max_row, max_time))
            x[1].resize((all_chunk_count + chunk_count, window_size, all_max_row))

        elif max_row < all_max_row:
            this_x[0].resize((chunk_count, window_size, all_max_row, max_time))
            this_x[1].resize((chunk_count, window_size, all_max_row))
            x[0].resize((all_chunk_count + chunk_count, window_size, all_max_row, max_time))
            x[1].resize((all_chunk_count + chunk_count, window_size, all_max_row))

        else:
            x[0].resize((all_chunk_count + chunk_count, window_size, all_max_row, max_time))
            x[1].resize((all_chunk_count + chunk_count, window_size, all_max_row))

        x[0][all_chunk_count:] = this_x[0]
        x[1][all_chunk_count:] = this_x[1]
        all_chunk_count += chunk_count

        labels = np.concatenate((labels, training_data['ref_sequence']))

    return prepare_dataset(x, labels)
