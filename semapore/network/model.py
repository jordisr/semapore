import tensorflow as tf
import numpy as np

def TimeDistributed2D(layer, name=None):
    return tf.keras.layers.TimeDistributed(tf.keras.layers.TimeDistributed(layer), name=name)

def SignalEmbedding(output_dim, params={}):

    conv1d = tf.keras.layers.Conv1D(filters=params.get('conv1d_filters', 32),
                                    kernel_size=params.get('conv1d_kernel_size', 3),
                                    strides=params.get('conv1d_strides', 5),
                                    activation='relu',
                                    padding="same",
                                   name= "Conv1D")

    rnn = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(output_dim//2, return_sequences=False))

    inputs = tf.keras.Input(shape=(None, 1))
    x = conv1d(inputs)
    outputs = rnn(x)

    return  tf.keras.Model(inputs=inputs, outputs=outputs)

def build_model(input_dim=128, encoder_dim=128, sequence_only=False):
    # if sequence_only, seq_embedding_dim = input_dim
    # otherwise, seq_embedding_dim = signal_embedding_dim = input_dim/2

    # convert tf.RaggedTensor inputs to tensors
    signal_input = tf.keras.Input(shape=(None, None, None, 1), ragged=True, name="SignalInput")
    signal_input_tensor = signal_input.to_tensor()

    alignment_input = tf.keras.Input(shape=(None, None), ragged=True, name="AlignmentInput")
    alignment_input_tensor = alignment_input.to_tensor()

    # embedding for raw signal features
    # input: (batch_size, num_columns, num_rows, max_time, 1)
    # output: (batch_size, num_columns, num_rows, signal_embedding_dim)
    signal_embedding = TimeDistributed2D(SignalEmbedding(input_dim // 2), name="SignalEmbedding")

    # embedding for each character in the alignment
    # input: (batch_size, num_columns, num_rows)
    # output: (batch_size, num_columns, num_rows, seq_embedding_dim)
    char_embedding = tf.keras.layers.Embedding(input_dim=11, mask_zero=True, output_dim=(input_dim if sequence_only else input_dim // 2))
    char_embedding_td = tf.keras.layers.TimeDistributed(char_embedding, name="CharacterEmbedding")

    # encode and summarize column information
    # input: (batch_size, num_columns, num_rows, input_dim)
    # output: (batch_size, num_columns, encoder_dim)
    rnn_1 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(encoder_dim // 2, return_sequences=False))
    rnn_1_td = tf.keras.layers.TimeDistributed(rnn_1, name="ColumnEncoder")

    # encode row information
    # input: (batch_size, num_columns, encoder_dim)
    # output: (batch_size, num_columns, encoder_dim)
    rnn_2 = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(encoder_dim // 2, return_sequences=True), name="RowEncoder")

    # input: (batch_size, num_columns, encoder_dim)
    # output: (batch_size, num_columns, output_dim=5)
    dense = tf.keras.layers.Dense(5, name="DenseOutput")

    # put layers together
    if sequence_only:
        x = char_embedding_td(alignment_input_tensor)
    else:
        x1 = signal_embedding(signal_input_tensor)
        x2 = char_embedding_td(alignment_input_tensor)
        x = tf.concat([x1, x2], axis=3)
    x = rnn_1_td(x)
    x = rnn_2(x)
    outputs = dense(x)

    model_name = ("sequence" if sequence_only else "sequence_signal")
    model = tf.keras.Model(inputs=[signal_input, alignment_input], outputs=outputs, name=model_name)

    return model
