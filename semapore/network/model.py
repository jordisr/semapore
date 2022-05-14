import tensorflow as tf
import numpy as np

class EmptyLayer(tf.keras.layers.Layer):
    # layer to stop propagation of mask, not sure if needed
    def __init__(self, **kwargs):
        super(EmptyLayer, self).__init__(**kwargs)
        self.supports_masking = False

    def call(self, inputs):
        return inputs

    def compute_output_shape(self, input_shape):
        return input_shape

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

    mask = tf.keras.layers.Masking()
    clear_mask = EmptyLayer(name="clear")

    inputs = tf.keras.Input(shape=(None, 1))
    x = conv1d(inputs)
    x = mask(x)
    x = rnn(x)
    x = clear_mask(x)

    return  tf.keras.Model(inputs=inputs, outputs=x)

def RnnModel(input_dim, output_dim, num_layers=1, reduce_dim=True, name=None):
    inputs = tf.keras.Input(shape=(None, input_dim))
    x = inputs
    if reduce_dim:
        for i in range(num_layers-1):
            x =  tf.keras.layers.Bidirectional(tf.keras.layers.GRU(output_dim // 2, return_sequences=True))(x)
        x = tf.keras.layers.Bidirectional(tf.keras.layers.GRU(output_dim // 2, return_sequences=False))(x)
    else:
        for i in range(num_layers):
            x =  tf.keras.layers.Bidirectional(tf.keras.layers.GRU(output_dim // 2, return_sequences=True))(x)
    
    return tf.keras.Model(inputs=inputs, outputs=x, name=name)

def build_model(seq_dim=64, signal_dim=64, encoder_dim=128, use_sequence=True, use_signal=False, use_draft=False, num_row_layers=1, num_col_layers=1):
    assert use_sequence or use_signal or use_draft, "Must specify at least one input"

    # convert tf.RaggedTensor inputs to tensors
    # tf.ensure_shape seems to be needed here since it's no longer in the de-serialization
    # explicitly including batch dimension in tf.ensure_shape, unlike tf.keras.Input
    signal_input = tf.keras.Input(shape=(None, None, None, 1), ragged=True, name="SignalInput")
    signal_input_tensor = tf.ensure_shape(signal_input.to_tensor(), (None, None, None, None, 1))

    alignment_input = tf.keras.Input(shape=(None, None), ragged=True, name="AlignmentInput")
    alignment_input_tensor = tf.ensure_shape(alignment_input.to_tensor(), (None, None, None))

    draft_input = tf.keras.Input(shape=(1, None), ragged=True, name="DraftInput")
    draft_input_tensor = tf.squeeze(draft_input, axis=1).to_tensor()

    # embedding for raw signal features
    # input: (batch_size, num_columns, num_rows, max_time, 1)
    # output: (batch_size, num_columns, num_rows, signal_embedding_dim)
    signal_embedding = TimeDistributed2D(SignalEmbedding(signal_dim), name="SignalEmbedding")

    # embedding for each character in the alignment (same embedding used for draft)
    # input: (batch_size, num_columns, num_rows)
    # output: (batch_size, num_columns, num_rows, seq_embedding_dim)
    char_embedding = tf.keras.layers.Embedding(input_dim=11, mask_zero=False, output_dim=seq_dim, name="CharacterEmbedding")

    # encode and summarize column information
    # input: (batch_size, num_columns, num_rows, input_dim)
    # output: (batch_size, num_columns, encoder_dim)
    col_embedding = RnnModel(input_dim=(signal_dim*use_signal + seq_dim*use_sequence), output_dim=encoder_dim, num_layers=num_col_layers, reduce_dim=True)
    col_embedding_td = tf.keras.layers.TimeDistributed(col_embedding, name="ColumnEncoder")

    # encode row information
    # input: (batch_size, num_columns, encoder_dim)
    # output: (batch_size, num_columns, encoder_dim)
    row_embedding = RnnModel(input_dim=encoder_dim, output_dim=encoder_dim, num_layers=num_row_layers, reduce_dim=False, name="RowEncoder")

    # input: (batch_size, num_columns, encoder_dim)
    # output: (batch_size, num_columns, output_dim=5)
    dense1 = tf.keras.layers.Dense(encoder_dim//2, name="DenseOutput1")
    dense2 = tf.keras.layers.Dense(5, name="DenseOutput2")

    # put layers together
    if use_signal:
        signal_embedding_output = signal_embedding([signal_input_tensor])

    if use_sequence:
        seq_embedding_output = char_embedding(alignment_input_tensor)

    if use_signal and use_sequence:
        x = tf.concat([signal_embedding_output, seq_embedding_output], axis=3)
    else:
        x = (signal_embedding_output if use_signal else seq_embedding_output)

    x = col_embedding_td(x)
    x = row_embedding(x)

    if use_draft:
        # draft concatenated in at the end
        draft_embedding = char_embedding(draft_input_tensor)
        x = tf.concat([x, draft_embedding], axis=2)
    
    x = dense1(x)
    outputs = dense2(x)

    model_name = ("draft_" if use_draft else "") + ("sequence_signal" if use_signal else "sequence")
    model = tf.keras.Model(inputs=[signal_input, alignment_input, draft_input], outputs=outputs, name=model_name)

    return model
