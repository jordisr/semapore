import os
import sys
import glob
import argparse

import numpy as np
from tqdm import tqdm
import tensorflow as tf

import semapore

def main():
    parser = argparse.ArgumentParser(description='Run polishing on multiple amplicon bins', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--pattern', help='Pattern to match directories', default=None)
    parser.add_argument('--names', help='List of directories', default=None)
    parser.add_argument('--base', help='Base directory with --names', default=None)
    parser.add_argument('-d','--draft', required=True, help='Draft assembly sequence (FASTA)', default=argparse.SUPPRESS)
    parser.add_argument('-a','--alignment', required=True, help='Reads aligned to draft assembly (BAM)', default=argparse.SUPPRESS)
    parser.add_argument('-r','--reads', required=True, help='Directory of basecalled reads (FAST5)', default=argparse.SUPPRESS)
    parser.add_argument('-t,','--threads', type=int, default=1, help='Number of threads for preprocessing')
    parser.add_argument('-m','--model', help='Neural network model')
    parser.add_argument('-w','--weights', help='Neural network weights (if using JSON for --model')
    parser.add_argument('--trimmed_reads', help='Trimmed reads used for assembly, needed if different than raw basecalls (FASTA/FASTQ)', default=argparse.SUPPRESS)
    parser.add_argument('--gpu', type=int, help='GPU to use for neural network acceleration')
    parser.add_argument('--nested_reads', action='store_true', help='Look for FAST5 reads in subfolders')
    parser.add_argument('--window_size', type=int, default=64, help='Draft window size')
    parser.add_argument('--max_time', type=int, default=80, help='Max number of signals per event')
    parser.add_argument('--just_draft', action='store_true', help='Just output draft sequence. For debugging.')

    args = parser.parse_args()

    if args.pattern:
        dirs = glob.glob(args.pattern)
    elif args.names:
        with open(args.names, 'r') as f:
            dirs = [os.path.join(args.base, d.strip('\n')) for d in f]
    else:
        sys.exit("Must provide list of directories with --pattern or --file")

    # load trained model
    if os.path.isdir(args.model):
        # load model from directory i.e. from Model.save()
        model = tf.keras.models.load_model(args.model, compile=False)
    else:
        # otherwise, load architecture from JSON file
        with open(args.model) as json_file:
            json_config = json_file.read()
            model = tf.keras.models.model_from_json(json_config, custom_objects={'EmptyLayer':semapore.network.EmptyLayer})

        # load trained model weights separately
        if os.path.isdir(args.weights):
            model_file = tf.train.latest_checkpoint(args.weights)
        else:
            model_file = args.weights
        model.load_weights(model_file)

    # decide on GPU
    if args.gpu:
        tf_device = '/gpu:{}'.format(args.gpu)
    else:
        gpu_list = tf.config.list_physical_devices('GPU')
        if len(gpu_list) > 0:
            tf_device = '/gpu:0'
        else: 
            tf_device = 'cpu'

    # load reads table
    if args.nested_reads:
        reads_table = semapore.util.find_nested_reads(args.reads)

    for dir_ in tqdm(dirs):
        # load draft sequence (seq also in pileup)
        if args.draft:
            (ref_id, ref_seq) = semapore.util.load_fastx(os.path.join(dir_, args.draft), "fasta")[0]

        # load alignment
        pileup = semapore.util.get_pileup_alignment(alignment=os.path.join(dir_, args.alignment), reference=os.path.join(dir_, args.draft))
        draft_id = pileup.columns[0]
        pileup_reads = list(pileup.columns)[1:]

        # load these reads
        if args.nested_reads:
            reads = semapore.util.get_reads(pileup_reads, paths=reads_table)
        else:
            reads = semapore.util.get_reads(pileup_reads, dir=args.reads)


        # load trimmed read sequences
        if args.trimmed_reads:
            semapore.util.trim_raw_reads(reads, os.path.join(dir_, args.trimmed_reads))

        # batch alignment and reads for inference
        sequence_input, signal_input, signal_input_mask, window_bounds = semapore.util.featurize_inputs(pileup, reads, window_size=args.window_size, max_time=args.max_time, draft_first=True)
        draft_input = sequence_input[:,:,0] 
        sequence_input = sequence_input[:,:,1:]

        with tf.device(tf_device):

            # TODO: parse stride from model configuration
            ds = semapore.network.dataset_from_arrays(signal_input, signal_input_mask, sequence_input, draft_input, stride=5)
            logits = []
            draft_labels = []
            for x in ds.batch(32):
                if args.just_draft:
                    rt = semapore.network.pileup_to_label(x[3])
                    for x1_ in rt:
                        for x2_ in x1_:
                            draft_labels.append(x2_)
                    #draft_labels.append(tf.ragged.map_flat_values(lambda x: x, rt))
                else:
                    logits.append(model.predict_on_batch(x))

        # viterbi decoding and stitch sequences
        if args.just_draft:
            decoded_seq = ''.join([''.join(np.take(np.array(['A','C','G','T']), x)) for x in draft_labels])
        else:
            logits = tf.concat(logits, axis=0)
            decoded_labels = semapore.network.greedy_decode(logits).numpy()
            decoded_seq = ''.join([''.join(np.take(np.array(['A','C','G','T']), x)) for x in decoded_labels])

        # output polish sequence in FASTA format
        print(semapore.util.fasta_format(os.path.basename(dir_), decoded_seq))


if __name__ == "__main__":
    main()
