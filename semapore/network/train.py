import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

def greedy_decode(logits):
    # doesn't merge repeated
    # returns a tf.RaggedTensor
    best_path = tf.cast(tf.math.argmax(logits, axis=2), np.int32)
    labels = tf.ragged.boolean_mask(best_path, best_path < 4)
    return labels

def pileup_to_label(x):
    x_ = x - 3
    return tf.ragged.boolean_mask(x_, x_ >= 0) % 4

def q_quality(x):
    return -10*np.log10(x)

def edit_distance(y_true, y_pred):
    y_true_sparse = y_true.to_sparse()
    predicted_labels = greedy_decode(y_pred).to_sparse()
    return tf.edit_distance(hypothesis=predicted_labels, truth=y_true_sparse, normalize=True)

def draft_edit_distance(draft, y_true):
    y_pred = tf.squeeze(pileup_to_label(draft), axis=1).to_sparse()
    return tf.edit_distance(hypothesis=y_pred, truth=y_true.to_sparse(), normalize=True)

def get_draft_edit_distance(ds):
    edit_distance = []
    for x,y in ds:
        edit_distance.append(tf.reduce_mean(draft_edit_distance(x[3],y)))
    edit_distance=np.array(edit_distance)
    return q_quality(np.mean(edit_distance))
    
def ctc_loss(ctc_merge_repeated=False):
    # closure for TF1 CTC loss
    # y_pred are unnormalized logits and y_true is a RaggedTensor of labels
    def loss(y_true, y_pred):
        y_true_sparse = tf.cast(y_true.to_sparse(),np.int32)
        sequence_length = tf.ones(tf.shape(y_pred)[0], dtype=np.int32)*tf.shape(y_pred)[1]
        return tf.compat.v1.nn.ctc_loss(inputs=y_pred,
                                        labels=y_true_sparse,
                                        sequence_length=sequence_length,
                                        time_major=False,
                                        preprocess_collapse_repeated=False,
                                        ctc_merge_repeated=ctc_merge_repeated,
                                        ignore_longer_outputs_than_inputs=True)
    return tf.function(func=loss)

def scst_ctc_loss(use_scst_loss=True, use_scst_baseline=True, scst_lambda=1, use_ml_loss=True, ctc_merge_repeated=False):
    # combine CTC loss with policy gradient as in Zhou 2018 (arxiv.org/abs/1712.07101)
    def loss_fn(y_true, y_pred):
        sequence_length = tf.ones(tf.shape(y_pred)[0], dtype=np.int32)*tf.shape(y_pred)[1]
        loss = 0
        y_true_sparse = y_true.to_sparse()
        if use_scst_loss:
            # sample single path from each output probability and convert to labels
            sampled_paths = tfp.distributions.Categorical(logits=y_pred).sample()
            sampled_labels_ragged = tf.ragged.boolean_mask(sampled_paths, sampled_paths != 4)
            sampled_labels = tf.dtypes.cast(sampled_labels_ragged.to_sparse(),tf.int32)

            reward_y_s = tf.clip_by_value(1 - tf.edit_distance(sampled_labels, y_true_sparse, normalize=True), clip_value_min=0, clip_value_max=1)

            # CTC loss for sampled label -log(P(y^s|x))
            negative_likelihood_ys = tf.compat.v1.nn.ctc_loss(inputs=y_pred,
                                            labels=sampled_labels,
                                            sequence_length=sequence_length,
                                            time_major=False,
                                            preprocess_collapse_repeated=False,
                                            ctc_merge_repeated=ctc_merge_repeated)

            if use_scst_baseline:
                # argmax labels
                argmax_labels = tf.math.argmax(y_pred, axis=2)
                argmax_labels = tf.ragged.boolean_mask(argmax_labels, argmax_labels != 4)
                argmax_labels = tf.dtypes.cast(argmax_labels.to_sparse(),tf.int32)
                reward_y_hat = tf.clip_by_value(1 - tf.edit_distance(argmax_labels, y_true_sparse, normalize=True), clip_value_min=0, clip_value_max=1)

                loss_scst = (reward_y_s - reward_y_hat)*negative_likelihood_ys
            else:
                loss_scst = reward_y_s * negative_likelihood_ys

            # SCST loss
            loss = loss + scst_lambda*loss_scst

        if use_ml_loss:
            # CTC loss for true label -log(P(y|x))
            negative_likelihood_y = tf.compat.v1.nn.ctc_loss(inputs=y_pred,
                                            labels=y_true_sparse,
                                            sequence_length=sequence_length,
                                            time_major=False,
                                            preprocess_collapse_repeated=False,
                                            ctc_merge_repeated=ctc_merge_repeated)

            # maximum likelihood loss
            loss += negative_likelihood_y
        return loss
    return loss_fn

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
