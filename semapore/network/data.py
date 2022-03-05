import os
import sys
import glob
import h5py
from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import mappy
import tensorflow as tf

import semapore.util

def get_alignment_coord(cs):
    """Parse CS string of mappy.Alignment
    """
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
                        r2q.append(coord_q)
                        q2r.append(coord_r)
                        coord_r += 1
                        coord_q += 1

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

def get_reference_sequences(draft_pileup, window_bounds, hit):
    """Get true labels by aligning draft to reference

    Args:
        draft_pileup (Series): draft column from pileup DataFrame
        window_bounds (ndarray): 2D array from featurize_inputs()
        aligner (mappy.Aligner): aligner to reference genome

    Returns:
        ndarray: draft_window_sequences, draft sequence strings
        ndarray: ref_window_sequences, ref sequence strings
        ndarray: ref_window_idx, positions with valid ref seq output
        str: ref_name, reference sequence that draft aligend to
    """

    # extract draft sequence from parsed pileup
    draft_sequence = ''.join([x[1] for x in draft_pileup if type(x) == tuple])

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

    print("Identity:{} Strand:{}".format(hit.mlen/hit.blen, hit.strand), file=sys.stderr)

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
            while type(draft_pileup_window[i]) is not tuple:
                i += 1
            draft_start = draft_pileup_window[i][0]

            # go bottom up, find first non insert character
            i = len(draft_pileup_window)-1
            while type(draft_pileup_window[i]) is not tuple:
                i -= 1
            draft_end = draft_pileup_window[i][0]
        except IndexError:
            # TODO: log these exceptions
            # skip sections with all inserts for draft sequence
            continue

        # check this chunk of draft contained in alignment to ref
        if draft_start >= hit.q_st and draft_end <= hit.q_en:
            ref_rel_start = q2r[draft_start-hit.q_st]
            ref_rel_end = q2r[draft_end-hit.q_st]

            # add one for inclusive ranges
            this_ref_seq = r_seq[ref_rel_start:ref_rel_end + 1]
            this_draft_seq = draft_sequence[draft_start:draft_end + 1]

            if len(this_ref_seq) > 0 and len(this_draft_seq) > 0 and ('N' not in this_ref_seq) and len(this_ref_seq) <= window_size:
                ref_window_idx.append(window_idx)
                ref_window_sequences.append(this_ref_seq)
                draft_window_sequences.append(this_draft_seq)
                #print("{}\t{}..{}\t{}..{}".format(hit.strand, this_draft_seq[:10],this_draft_seq[-10:], this_ref_seq[:10],this_ref_seq[-10:]), file=sys.stderr)

            else:
                print("Warning: window {}, draft_len={} ref_len={}".format(window_idx, len(this_draft_seq), len(this_ref_seq)),  file=sys.stderr)

    return np.array(draft_window_sequences), np.array(ref_window_sequences), np.array(ref_window_idx), np.array([ref_name, hit.strand, hit.mlen/hit.blen, hit.q_st, hit.q_en, hit.r_st, hit.r_en])

def make_training_data(draft, alignment, reads, hit=None, out="tmp", aligner=None, reference="", trimmed_reads="", window=64, max_time=80):
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
    pileup = semapore.util.get_pileup_alignment(alignment=alignment, reference=draft)
    draft_id = pileup.columns[0]
    draft_pileup = pileup[draft_id]
    pileup_reads = list(pileup.columns)[1:]

    # load reads
    if type(reads) is str:
        reads = semapore.util.get_reads(pileup_reads, dir=reads)
    elif type(reads) is dict:
        reads = semapore.util.get_reads(pileup_reads, paths=reads)
    else:
        sys.exit("No reads provided!")

    # load trimmed read sequences
    if trimmed_reads:
        semapore.util.trim_raw_reads(reads, trimmed_reads)

    # batch alignment and reads for inference
    sequence_input, signal_input, signal_length, signal_input_mask, window_bounds = semapore.util.featurize_inputs(pileup, reads, window_size=window, max_time=max_time, draft_first=True)
    draft_input = sequence_input[:,:,0] 
    sequence_input = sequence_input[:,:,1:]

    # initialize mappy Aligner
    #if reference and not aligner:
    #    aligner = mp.Aligner(reference, preset='map-ont', n_threads=1)

    print("Moving to draft: {}".format(draft), file=sys.stderr)
    draft_sequences, ref_sequences, ref_windows, ref_name = get_reference_sequences(draft_pileup, window_bounds, hit)

    draft_input = draft_input[ref_windows]
    sequence_input = sequence_input[ref_windows]
    signal_input = signal_input[ref_windows]
    signal_input_mask = signal_input_mask[ref_windows]
    files = np.array([draft, alignment, reference, trimmed_reads])

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

    # save windows where draft and reference aren't identitcal
    ref_is_diff =  ~(ref_sequences == draft_sequences)
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
            pass

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

def make_training_dir(pattern, draft, alignment, reads, reference, trimmed_reads="", out="training", window=64, max_time=80, threads=1):
    """Make training NPZ files for nested directory of assemblies

    Args:
        pattern (str): wildcard pattern to be expanded for directory names
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
    #aligner = mp.Aligner(reference, preset='map-ont', n_threads=1)
    #files = [(f, aligner) for f in glob.glob(pattern)]
    files = glob.glob(pattern)
    hits = []

    if threads > 1:
        aligner = mappy.Aligner(reference, preset='map-ont', n_threads=threads)
        training_input = []
        for f in files:
            draft_path = "{}/{}".format(f, draft)
            seqs = semapore.util.load_fastx(draft_path, "fasta")
            if len(seqs) > 0 and len(seqs[0][1]) > 0 :
                hit = AlignmentCopy(aligner, next(aligner.map(seqs[0][1], cs=True)))
                training_input.append((f, hit))

        #print(len(training_input))
        training_helper = TrainingDataHelper(
                            out=out,
                            reference=reference,
                            draft=draft,
                            alignment=alignment,
                            trimmed_reads=trimmed_reads,
                            reads=reads,
                            window=window,
                            max_time=max_time)

        with Pool(processes=threads) as pool:
            pool.map(training_helper, training_input, chunksize=20)
    else:
        aligner = mappy.Aligner(reference, preset='map-ont')

        for f in files:
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

def ragged_from_list_of_lists(l):
    return tf.RaggedTensor.from_row_lengths(np.concatenate(l), np.array([len(i) for i in l]))

def load_npz_data(f):
    training_data = np.load(f)
    x = [training_data['signal_input'], training_data['sequence_input'].astype(np.int32)]
    labels = ragged_from_list_of_lists([[{'A':0,'C':1,'G':2,'T':3}[i] for i in j] for j in training_data['ref_sequence']])
    return x, labels

def prepare_dataset(x, labels):
    x_ds = tuple([tf.data.Dataset.from_tensor_slices(x_) for x_ in x])

    # what to do about N nucleotide? N > A
    labels_ds = tf.data.Dataset.from_tensor_slices(tf.cast(
        ragged_from_list_of_lists([[{'A':0,'C':1,'G':2,'T':3, 'N':0}[i] for i in j] for j in labels]), np.int32))

    return tf.data.Dataset.zip((x_ds, labels_ds))

def dataset_from_files(files, ndarray_out=False):
    """Merge and pad NPZ data and convert to tf.data.Dataset in memory

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

    if ndarray_out:
        return x, labels
    else:
        return prepare_dataset(x, labels)

def generator_dataset_from_files(files):
    """Create dataset from generator

    Args:
        files (list): *.npz files created with make_training_dir()/make_training_data()

    Returns:
        tf.data.Dataset: dataset of zipped inputs and labels
    """

    def npz_gen(files_):
        for f in files_:
            training_data = np.load(f)

            x1 = training_data['signal_input'].astype(np.float32)
            x2 = training_data['sequence_input'].astype(np.int32)
            x3 = training_data['draft_input'].astype(np.int32)
            labels = training_data['ref_sequences']

            for d in zip(x1,x2,x3,labels):
                x1_ = tf.RaggedTensor.from_tensor(tf.expand_dims(d[0], axis=3), ragged_rank=2)
                x2_ = tf.RaggedTensor.from_tensor(d[1])
                x3_ = tf.RaggedTensor.from_tensor(tf.expand_dims(d[2], axis=0))

                label =[{'A':0,'C':1,'G':2,'T':3}[i] for i in d[-1]]
                label = tf.cast(tf.RaggedTensor.from_row_lengths(label, row_lengths=[len(label)]), tf.int32)

                yield ((x1_, x2_, x3_), label)

    ds = tf.data.Dataset.from_generator(npz_gen,
                                         args=[files],
                                         output_signature=((tf.RaggedTensorSpec(shape=(None,None,None, 1), ragged_rank=2, dtype=tf.float32),
                                                            tf.RaggedTensorSpec(shape=(None,None), dtype=tf.int32),
                                                            tf.RaggedTensorSpec(shape=(1,None), ragged_rank=1, dtype=tf.int32)
                                                            ),
                                                            tf.RaggedTensorSpec(shape=(1,None), ragged_rank=1, dtype=tf.int32)))

    return ds

def generator_dataset_from_hdf5(files):
    """Create dataset from generator

    Args:
        files (list): *.hdf5 files

    Returns:
        tf.data.Dataset: dataset of zipped inputs and labels
    """

    def hdf5_gen(files_):
        for f in files_:
            h5 = h5py.File(f, 'r')
            for group in h5:
                training_data = h5[group]
                x1 = np.array(training_data['signal_input'], dtype=np.float32)
                x2 = np.array(training_data['sequence_input'], dtype=np.int32)
                x3 = np.array(training_data['draft_input'], dtype=np.int32)
                labels = np.array(training_data['ref_sequences'])

                for d in zip(x1,x2,x3,labels):
                    x1_ = tf.RaggedTensor.from_tensor(tf.expand_dims(d[0], axis=3), ragged_rank=2)
                    x2_ = tf.RaggedTensor.from_tensor(d[1])
                    x3_ = tf.RaggedTensor.from_tensor(tf.expand_dims(d[2], axis=0))
                    label =[{'A':0,'C':1,'G':2,'T':3}[i] for i in d[-1].decode('utf-8')]
                    label = tf.cast(tf.RaggedTensor.from_row_lengths(label, row_lengths=[len(label)]), tf.int32)

                    yield ((x1_, x2_, x3_), label)

    ds = tf.data.Dataset.from_generator(hdf5_gen,
                                         args=[files],
                                         output_signature=((tf.RaggedTensorSpec(shape=(None,None,None, 1), ragged_rank=2, dtype=tf.float32),
                                                            tf.RaggedTensorSpec(shape=(None,None), dtype=tf.int32),
                                                            tf.RaggedTensorSpec(shape=(1,None), ragged_rank=1, dtype=tf.int32)
                                                            ),
                                                            tf.RaggedTensorSpec(shape=(1,None), ragged_rank=1, dtype=tf.int32)))

    return ds

def merge_npz_to_hdf5(npz_files, name, target_size=100, compression=None):
    numeric_keys = ['draft_input', 'sequence_input', 'signal_input', 'signal_mask']
    string_keys = ['files', 'ref_sequences', 'draft_sequences', 'ref_name']
    
    file_sizes = np.array([os.path.getsize(x)/1024**2 for x in npz_files])
    size_bins = np.floor(np.cumsum(file_sizes) / target_size).astype(int)
    size_bin_end = np.unique(size_bins, return_index=True)[1][1:]
    num_digits = len(str(size_bins[-1]))
    print("Merging {} NPZ files into {} HDF5 files...".format(len(file_sizes), size_bins[-1]), file=sys.stderr)
    
    range_start = 0
    for i,range_end in enumerate(tqdm(size_bin_end)):
        h5_path = "{name}.{num:0{num_digits}}.hdf5".format(name=name, num=i, num_digits=num_digits)
        with h5py.File(h5_path, "w") as h5_f:
            for f in npz_files[range_start:range_end]:
                grp = h5_f.create_group(os.path.basename(f))
                with np.load(f) as data:
                    for k in numeric_keys:
                        grp.create_dataset(k, data=data[k], compression=compression)
                    for k in string_keys:
                        grp.create_dataset(k, data=list(data[k]), dtype=h5py.string_dtype(encoding='utf-8'), compression=compression)
        range_start = range_end

def feature_from_tensor(tensor):
    # for serialization into TFRecords
    serialized_tensor = tf.io.serialize_tensor(tensor)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_tensor.numpy()]))

def write_serialized_examples_from_hdf5(files):
    # writes one TFRecord file per HDF5 file
    # TODO: combine with merge_npz_to_hdf5 to skip generation of HDF5 intermediates
    for f in files:
        out_path = "{}.tfrecord".format(os.path.splitext(f)[0])
        with tf.io.TFRecordWriter(out_path) as file_writer:
            h5 = h5py.File(f, 'r')
            for group in h5:
                training_data = h5[group]
                x1 = training_data['signal_input'].astype(np.float32)
                x2 = training_data['sequence_input'].astype(np.int32)
                x3 = training_data['draft_input'].astype(np.int32)
                labels = training_data['ref_sequences']

                for x1_, x2_, x3_, labels_ in zip(x1,x2,x3,labels):

                    label_values = [{'A':0,'C':1,'G':2,'T':3}[i] for i in labels_.decode('utf-8')]
                    label_length = [len(label_values)]

                    features_for_example = {"signal_input": feature_from_tensor(x1_),
                                            "sequence_input": feature_from_tensor(x2_),
                                            "draft_input": feature_from_tensor(x3_),
                                            "label_values": feature_from_tensor(label_values),
                                            "label_lengths": feature_from_tensor(label_length)}

                    example = tf.train.Example(features=tf.train.Features(feature=features_for_example))
                    file_writer.write(example.SerializeToString())

def make_decoder(stride=1):
    @tf.function
    def decode_tfrecord(x):
        # map over TFRecordDataset to get tensors
        feature_schema = {"signal_input": tf.io.FixedLenFeature([], dtype=tf.string),
                        "signal_mask": tf.io.FixedLenFeature([], dtype=tf.string),
                        "sequence_input": tf.io.FixedLenFeature([], dtype=tf.string),
                        "draft_input": tf.io.FixedLenFeature([], dtype=tf.string),
                        "label_values": tf.io.FixedLenFeature([], dtype=tf.string),
                        "label_lengths": tf.io.FixedLenFeature([], dtype=tf.string)}
        
        parsed_example = tf.io.parse_single_example(x, features=feature_schema)
        
        signal_tensor = tf.ensure_shape(tf.io.parse_tensor(parsed_example['signal_input'], tf.float32), (None, None, None))
        signal_mask_tensor = tf.ensure_shape(tf.io.parse_tensor(parsed_example['signal_mask'], tf.bool), (None, None,None))
        sequence_tensor = tf.ensure_shape(tf.io.parse_tensor(parsed_example['sequence_input'], tf.int32), (None, None))
        draft_tensor = tf.io.parse_tensor(parsed_example['draft_input'], tf.int32)
        labels = tf.ensure_shape(tf.io.parse_tensor(parsed_example['label_values'], tf.int32), (None,))
        
        signal_tensor = tf.RaggedTensor.from_tensor(tf.expand_dims(signal_tensor, axis=3), ragged_rank=2)
        signal_mask_tensor = tf.RaggedTensor.from_tensor(signal_mask_tensor, ragged_rank=1)
        sequence_tensor = tf.RaggedTensor.from_tensor(sequence_tensor)
        draft_tensor = tf.RaggedTensor.from_tensor(tf.expand_dims(draft_tensor, axis=0))

        return ((signal_tensor, signal_mask_tensor[:,:,::stride], sequence_tensor, draft_tensor), labels)
    return decode_tfrecord

def dataset_from_arrays(signal, signal_mask, sequence, draft):
    # takes ndarrays returned by featurize_inputs()
    # return a tf.data.Dataset
    # ndarray -> RaggedTensor -> Tensor wasn't working for forward pass (to_tensor() raising error)
    # generator ensures correct types for input to network

    def gen(signal, signal_mask, sequence, draft):
        for x in zip(signal, signal_mask, sequence, draft):
            x1_ = tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(x[0], tf.float32), axis=3), ragged_rank=2)
            x2_ = tf.RaggedTensor.from_tensor(tf.cast(x[1], tf.bool), ragged_rank=2)
            x3_ = tf.RaggedTensor.from_tensor(tf.cast(x[2], tf.int32))
            x4_ = tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(x[3], tf.int32), axis=0))

            yield x1_, x2_, x3_, x4_

    ds = tf.data.Dataset.from_generator(gen,
                                         args=[signal, signal_mask, sequence, draft],
                                         output_signature=((tf.RaggedTensorSpec(shape=(None,None,None, 1), ragged_rank=2, dtype=tf.float32),
                                                            tf.RaggedTensorSpec(shape=(None,None,None), ragged_rank=2, dtype=tf.bool),
                                                            tf.RaggedTensorSpec(shape=(None,None), dtype=tf.int32),
                                                            tf.RaggedTensorSpec(shape=(1,None), ragged_rank=1, dtype=tf.int32)
                                                            )))

    return ds

def write_serialized_from_npz(npz_files, name, num_files=10):    
    file_sizes = np.array([os.path.getsize(x)/1024**2 for x in npz_files])
    size_bins = np.floor(np.cumsum(file_sizes) / num_files).astype(int)
    size_bin_end = np.unique(size_bins, return_index=True)[1][1:]
    num_digits = len(str(size_bins[-1]))
    
    range_start = 0
    for i,range_end in enumerate(tqdm(size_bin_end)):
        out_path = "{name}.{num:0{num_digits}}.tfrecord".format(name=name, num=i, num_digits=num_digits)
        with tf.io.TFRecordWriter(out_path) as file_writer:
            for f in npz_files[range_start:range_end]:                
                with np.load(f) as data:
                    x1 = data['signal_input'].astype(np.float32)
                    x2 = data['signal_mask'].astype(bool)
                    x3 = data['sequence_input'].astype(np.int32)
                    x4 = data['draft_input'].astype(np.int32)
                    labels = data['ref_sequences']

                    for x1_, x2_, x3_, x4_, labels_ in zip(x1,x2,x3,x4,labels):

                        label_values = [{'A':0,'C':1,'G':2,'T':3}[i] for i in labels_]
                        label_length = [len(label_values)]

                        features_for_example = {"signal_input": feature_from_tensor(x1_),
                                                "signal_mask": feature_from_tensor(x2_),
                                                "sequence_input": feature_from_tensor(x3_),
                                                "draft_input": feature_from_tensor(x4_),
                                                "label_values": feature_from_tensor(label_values),
                                                "label_lengths": feature_from_tensor(label_length)}

                        example = tf.train.Example(features=tf.train.Features(feature=features_for_example))
                        file_writer.write(example.SerializeToString())
        range_start = range_end