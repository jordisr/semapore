from tqdm import tqdm
import numpy as np
import tensorflow as tf

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

def feature_from_tensor(tensor):
    # for serialization into TFRecords
    serialized_tensor = tf.io.serialize_tensor(tensor)
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[serialized_tensor.numpy()]))

def write_serialized_examples(examples, outfile):
    with tf.io.TFRecordWriter(outfile) as file_writer:        
        for example in tqdm(examples):               
            features_for_tfexample = {k:feature_from_tensor(v) for k,v in example.items()}
            tfexample = tf.train.Example(features=tf.train.Features(feature=features_for_tfexample))
            file_writer.write(tfexample.SerializeToString())

def decode_tfrecord(x):
    # map over TFRecordDataset to get tensors
    feature_schema = {"signal_values": tf.io.FixedLenFeature([], dtype=tf.string),
                    "signal_row_lengths": tf.io.FixedLenFeature([], dtype=tf.string),
                    "signal_col_lengths": tf.io.FixedLenFeature([], dtype=tf.string),
                    "sequence_values": tf.io.FixedLenFeature([], dtype=tf.string),
                    "sequence_col_lengths": tf.io.FixedLenFeature([], dtype=tf.string),
                    "draft": tf.io.FixedLenFeature([], dtype=tf.string),
                    "labels": tf.io.FixedLenFeature([], dtype=tf.string)
                     }

    parsed_example = tf.io.parse_single_example(x, features=feature_schema)

    signal_values = tf.io.parse_tensor(parsed_example['signal_values'], tf.int16)
    signal_row_lengths = tf.io.parse_tensor(parsed_example['signal_row_lengths'], tf.int32)
    signal_col_lengths = tf.io.parse_tensor(parsed_example['signal_col_lengths'], tf.int32)
    sequence_values = tf.io.parse_tensor(parsed_example['sequence_values'], tf.int16)
    sequence_col_lengths = tf.io.parse_tensor(parsed_example['sequence_col_lengths'], tf.int32)
    draft = tf.io.parse_tensor(parsed_example['draft'], tf.int16)
    labels = tf.ensure_shape(tf.io.parse_tensor(parsed_example['labels'], tf.int16), (None,))
    
    signal_rt = tf.RaggedTensor.from_row_lengths(
                    values=tf.RaggedTensor.from_row_lengths(values=tf.expand_dims(signal_values, 1), row_lengths=signal_row_lengths), 
                    row_lengths=signal_col_lengths)
    
    sequence_rt = tf.RaggedTensor.from_row_lengths(values=sequence_values, row_lengths=sequence_col_lengths)
    
    draft_rt = tf.RaggedTensor.from_tensor(tf.expand_dims(draft, axis=0))

    return ((signal_rt, sequence_rt, draft_rt), labels)