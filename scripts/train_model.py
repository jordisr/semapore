import os
import sys
import glob
import argparse
import datetime
import subprocess
import numpy as np
import tensorflow as tf
import wandb
from wandb.keras import WandbCallback
import semapore

class EditDistance(tf.keras.metrics.Metric):
    def __init__(self, name='edit_distance', **kwargs):
        super(EditDistance, self).__init__(name=name, **kwargs)
        self.edit_distance = self.add_weight(name='edit_distance', initializer='zeros')
        self.num_examples = self.add_weight(name='num_examples', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.num_examples.assign_add(1)
        self.edit_distance.assign_add(tf.reduce_mean(semapore.network.edit_distance(y_true=y_true, y_pred=y_pred)))

    def result(self):
        return self.edit_distance/self.num_examples

    def reset_state(self):
        self.edit_distance.assign(0)
        self.num_examples.assign(0)

def train_model(args):

    # set random seed
    if args.seed is None:
        args.seed = int(datetime.datetime.now().timestamp())
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)

    config = {}

    # get current git commit
    repo_path = os.path.dirname(os.path.realpath(__file__))
    current_commit = subprocess.check_output(["git","-C",repo_path,"rev-parse", "HEAD"]).rstrip()
    config['commit'] = current_commit

    # initialize W&B run
    # TODO: use single config dict that is passed to W&B
    if args.wandb:
        wandb.init(project="semapore", entity="jordisr", name=args.name)
        wandb_log = ['epochs',
                    'seed',
                    'batch_size',
                    'validation_size', 
                    'ctc_merge_repeated',
                    'optimizer',
                    'learning_rate',
                    'seq_dim', 
                    'signal_dim',
                    'encoder_dim',
                    'loss',
                    'policy_lambda',
                    'policy_skip_epochs',
                    'architecture',
                    'set_error_fraction']
        wandb_config = {}
        for param in wandb_log:
            wandb_config[param] = getattr(args, param)
        for k,v in config.items():
            wandb_config[k] = v
        wandb.config.update(wandb_config)
        run_name = wandb.run.name
    else:
        if args.name:
            run_name = args.architecture + '_' + args.name
        else:
            run_name = args.architecture
    
    # directory for model checkpoints and logging
    out_dir = "{}.{}".format(run_name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M"))
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    # log all command line arguments
    log_file = open(out_dir+'/train.log','w')
    print(' '.join(sys.argv),file=log_file)
    print('Command-line arguments:',file=log_file)
    for k,v in args.__dict__.items():
        print(k,'=',v, file=log_file)

    # allow memory growth when using GPU
    gpu_devices = tf.config.list_physical_devices('GPU')
    for d in gpu_devices:
        tf.config.experimental.set_memory_growth(d, True)

    if len(args.gpu) < 1:
        if len(gpu_devices) > 0:
            train_devices = list(map(lambda x:"/gpu:{}".format(x.name.split(':')[2]), gpu_devices))
        else:
            train_devices = ["/cpu:0"]
    else:
        train_devices = list(map(lambda x:"/gpu:{}".format(x), args.gpu))

    if len(train_devices) > 1:
        strategy = tf.distribute.MirroredStrategy(devices=train_devices)
    else:
        strategy = tf.distribute.OneDeviceStrategy(device=train_devices[0])
    print("Number of devices: {}".format(strategy.num_replicas_in_sync))

    # load training datas
    decoder_fn = semapore.network.decode_tfrecord

    # mix training examples with and without errors
    if (args.set_error_fraction > 0) and (args.set_error_fraction <=1):
        training_files_errors = glob.glob(os.path.join(args.data, "errors.*.tfrecord"))    
        errors_dataset = tf.data.TFRecordDataset(training_files_errors).map(decoder_fn)
        print("Found {} files for training, sampling examples with errors at rate of {}".format(len(training_files_errors), args.set_error_fraction))

        if args.set_error_fraction == 1:
            dataset = errors_dataset
        else:
            training_files_noerrors = glob.glob(os.path.join(args.data, "same.*.tfrecord"))
            noerrors_dataset = tf.data.TFRecordDataset(training_files_noerrors).map(decoder_fn)
            dataset = tf.data.Dataset.sample_from_datasets([errors_dataset, noerrors_dataset], 
                                                            weights=[args.set_error_fraction, 1-args.set_error_fraction],
                                                            stop_on_empty_dataset=True)
    else:
        if os.path.isdir(args.data):
            training_files = glob.glob("{}/*.tfrecord".format(args.data))
        else:
            training_files = [args.data]
        dataset = tf.data.TFRecordDataset(training_files).map(decoder_fn)

    dataset = dataset.shuffle(buffer_size=500, reshuffle_each_iteration=True)
    batched_dataset = dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=args.batch_size, drop_remainder=True))
    
    if args.validation:
        if os.path.isdir(args.validation):
            validation_files = glob.glob("{}/*.tfrecord".format(args.validation))
        else:
            validation_files = [args.validation]

        validation_dataset = tf.data.TFRecordDataset(validation_files).map(decoder_fn)
        validation_dataset = validation_dataset.apply(tf.data.experimental.dense_to_ragged_batch(batch_size=args.batch_size, drop_remainder=True))
        train_dataset = batched_dataset
    else:
        train_dataset = batched_dataset.skip(args.validation_size)
        validation_dataset = batched_dataset.take(args.validation_size)

    if args.batches > 0:
        train_dataset = train_dataset.take(args.batches)

    # parse architecture from argument
    architecture_args = args.architecture.split('_')
    use_draft = (architecture_args[0] == 'draft')
    use_signal = (architecture_args[-1] == 'signal')
    use_sequence = ('sequence' in architecture_args)

    # parse loss function from argument
    loss_args = args.loss.split('_')
    use_ml_loss=('ml' in loss_args) 
    use_scst_loss= ('policy' in loss_args)
    use_scst_baseline=('baseline' in loss_args)

    with strategy.scope():
        # get the neural network architecture model
        model = semapore.network.build_model(
            seq_dim=args.seq_dim, 
            signal_dim=args.signal_dim, 
            encoder_dim=args.encoder_dim,
            use_signal=use_signal,
            use_draft=use_draft,
            use_sequence=use_sequence
        )

        # save architecture
        json_config = model.to_json()
        with open(out_dir+'/model.json', 'w') as json_file:
            json_file.write(json_config)

        # restart training from weights in checkpoint
        if args.restart:
            if os.path.isdir(args.restart):
                model_file = tf.train.latest_checkpoint(args.restart)
            else:
                model_file = args.restart
            model.load_weights(model_file)

        log_file.close()

        if args.optimizer == 'Adam':
            optimizer = tf.keras.optimizers.Adam(args.learning_rate)
        else:
            optimizer = tf.keras.optimizers.SGD(args.learning_rate)

        # callbacks for training
        os.makedirs(os.path.join(out_dir, "checkpoints"))
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(os.path.join(out_dir, "checkpoints", "{epoch:02d}.hdf5"), save_freq='epoch', save_weights_only=True)
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(out_dir,'logs'), update_freq='epoch')
        #tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=os.path.join(out_dir,'logs'), update_freq=50, profile_batch='50,100')
        terminante_on_nan_callback = tf.keras.callbacks.TerminateOnNaN()
        csv_logger_callback = tf.keras.callbacks.CSVLogger(os.path.join(out_dir,'train.csv'), separator=',', append=False)

        callbacks = [model_checkpoint_callback,
                    tensorboard_callback,
                    terminante_on_nan_callback,
                    csv_logger_callback]

        if args.early_stopping:
            early_stopping_callback = tf.keras.callbacks.EarlyStopping(monitor=args.early_stopping, min_delta=0, patience=10, verbose=1, restore_best_weights=True)
            callbacks.append(early_stopping_callback)

        if args.wandb:
            callbacks.append(WandbCallback())

        epochs_to_train = args.epochs
        if use_scst_loss and args.policy_skip_epochs > 0:
            assert(args.policy_skip_epochs < args.epochs)

            loss_fn = semapore.network.scst_ctc_loss(use_scst_loss=False, 
                                        use_scst_baseline=False,
                                        use_ml_loss=True, 
                                        ctc_merge_repeated=args.ctc_merge_repeated)

            model.compile(optimizer=optimizer, loss=loss_fn, metrics=[EditDistance()])
            model.fit(train_dataset, epochs=args.policy_skip_epochs, validation_data=validation_dataset, callbacks=callbacks)
            epochs_to_train -= args.policy_skip_epochs

        loss_fn = semapore.network.scst_ctc_loss(use_scst_loss=use_scst_loss, 
                                                use_scst_baseline=use_scst_baseline,
                                                scst_lambda=args.policy_lambda, 
                                                use_ml_loss=use_ml_loss, 
                                                ctc_merge_repeated=args.ctc_merge_repeated)

        model.compile(optimizer=optimizer, loss=loss_fn, metrics=[EditDistance()])

        model.fit(train_dataset, epochs=epochs_to_train, validation_data=validation_dataset, callbacks=callbacks)

        model.save(os.path.join(out_dir,"final_model"), include_optimizer=False)

if __name__ == '__main__':
    # general options
    parser = argparse.ArgumentParser(description='Train new Semapore model')
    parser.add_argument('--data', help='Path to training files in serialized TFRecord format', required=True)
    parser.add_argument('--name', default=None, help='Name of run')
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train on')
    parser.add_argument('--gpu', nargs="+", default=[], help='Specify which GPU devices for training (default: train on all available GPUs)')
    parser.add_argument('--restart', default=False, help='Trained model to load (if directory, loads latest from checkpoint file)')
    parser.add_argument('--seed', type=int, default=None, help='Explicitly set random seed')
    parser.add_argument('--wandb', action='store_true', help='Log run on Weights & Biases')
    parser.add_argument('--set_error_fraction', type=float, default=0, help='Set ratio of examples with/without errors in draft sequence')
    parser.add_argument('--batches', type=int, default=0, help='Only train for N batches each epoch')    

    # training options
    parser.add_argument('--batch_size', default=64, type=int, help='Minibatch size for training')
    parser.add_argument('--validation', default=None, help="Path to directory with one or more TFRecord files")
    parser.add_argument('--validation_size', default=100, type=int, help='Number of batches to withold for validation set (if --validation not specified)')
    parser.add_argument('--ctc_merge_repeated', action='store_true', default=False, help='boolean option for tf.compat.v1.nn.ctc_loss')
    parser.add_argument('--optimizer', default="Adam", choices=["Adam", "SGD"], help='Optimizer for gradient descent')
    parser.add_argument('--early_stopping', default=None, choices=["val_loss", "val_edit_distance"], help='Stop training when metric stops decreasing')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
    parser.add_argument('--loss', choices=['ml','ml_policy','ml_policy_baseline','policy_baseline','policy'], default='ml', help='Loss function for training')
    parser.add_argument('--policy_lambda', type=float, default=1, help='Weight given to policy gradient loss')
    parser.add_argument('--policy_skip_epochs', type=int, default=0, help='Only start policy gradient after first N epochs')
    
    # architecture options
    parser.add_argument('--architecture', choices=['sequence','signal','draft_signal','sequence_signal','draft_sequence','draft_sequence_signal'], default='sequence', help='')
    parser.add_argument('--seq_dim', default=32, type=int, help='')
    parser.add_argument('--signal_dim', default=128, type=int, help='')
    parser.add_argument('--encoder_dim', default=512, type=int, help='')
    parser.add_argument('--num_row_layers', default=1, type=int, help='')
    parser.add_argument('--num_col_layers', default=3, type=int, help='')
    
    args = parser.parse_args()
    train_model(args)