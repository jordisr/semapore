import tensorflow as tf
import numpy as np

def greedy_decode(logits):
    # doesn't merge repeated
    # returns a tf.RaggedTensor
    best_path = tf.cast(tf.math.argmax(logits, axis=2), np.int32)
    labels = tf.ragged.boolean_mask(best_path, best_path < 4)
    return labels

def pileup_to_label(x):
    x_ = x - 3
    return tf.ragged.boolean_mask(x_, x_ >= 0) % 4

def edit_distance(y_true, y_pred):
    y_true_sparse = y_true.to_sparse()
    predicted_labels = greedy_decode(y_pred).to_sparse()
    return tf.edit_distance(hypothesis=predicted_labels, truth=y_true_sparse, normalize=True)

def ctc_loss(ctc_merge_repeated=False):
    # closure for TF1 CTC loss
    # y_pred are unnormalized logits and y_true is a RaggedTensor of labels
    def loss(y_true, y_pred):
        y_true_sparse = y_true.to_sparse()
        sequence_length = tf.ones(tf.shape(y_pred)[0], dtype=np.int32)*tf.shape(y_pred)[1]
        return tf.compat.v1.nn.ctc_loss(inputs=y_pred,
                                        labels=y_true_sparse,
                                        sequence_length=sequence_length,
                                        time_major=False,
                                        preprocess_collapse_repeated=False,
                                        ctc_merge_repeated=ctc_merge_repeated,
                                        ignore_longer_outputs_than_inputs=True)
    return tf.function(func=loss)

def training_loop(model, dataset, optimizer=tf.keras.optimizers.Adam(), epochs=1, log_frequency=10, ctc_merge_repeated=False):
    # minimal training loop for testing
    for epoch in range(epochs):
        for t,(X,y) in dataset.enumerate():

            with tf.GradientTape() as tape:
                y_pred = model(X)

                loss = tf.reduce_mean(tf.compat.v1.nn.ctc_loss(inputs=y_pred,
                                                labels=y.to_sparse(),
                                                sequence_length=tf.ones(tf.shape(y_pred)[0], dtype=np.int32)*tf.shape(y_pred)[1],
                                                time_major=False,
                                                preprocess_collapse_repeated=False,
                                                ctc_merge_repeated=ctc_merge_repeated,
                                                ignore_longer_outputs_than_inputs=True))

            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            if t % log_frequency == 0:
                print("Epoch:{}\tIteration:{}\tLoss:{}".format(epoch, t,loss.numpy()))
