import os
import sys
import h5py
from tqdm import tqdm
import numpy as np
import tensorflow as tf

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

def dataset_from_arrays(signal, signal_mask, sequence, draft, stride=1):
    # takes ndarrays returned by featurize_inputs()
    # return a tf.data.Dataset
    # ndarray -> RaggedTensor -> Tensor wasn't working for forward pass (to_tensor() raising error)
    # generator ensures correct types for input to network

    def gen(signal, signal_mask, sequence, draft):
        for x in zip(signal, signal_mask, sequence, draft):
            x1_ = tf.RaggedTensor.from_tensor(tf.expand_dims(tf.cast(x[0], tf.float32), axis=3), ragged_rank=2)
            x2_ = tf.RaggedTensor.from_tensor(tf.cast(x[1][:,:,::stride], tf.bool), ragged_rank=2)
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
    size_bins = np.floor(np.cumsum(file_sizes) / (np.sum(file_sizes)/num_files)).astype(int)
    size_bin_end = np.unique(size_bins, return_index=True)[1][1:]
    num_digits = len(str(size_bins[-1]-1)) # files are 0-indexed i.e. only need 1 digit for 0-9
    
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