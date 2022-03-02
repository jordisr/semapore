import os
import sys
import argparse
import numpy as np
import tensorflow as tf
import semapore

def main():
    parser = argparse.ArgumentParser(description='Neural polishing from signal', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d','--draft', required=True, help='Draft assembly sequence (FASTA)', default=argparse.SUPPRESS)
    parser.add_argument('-a','--alignment', required=True, help='Reads aligned to draft assembly (BAM)', default=argparse.SUPPRESS)
    parser.add_argument('-r','--reads', required=True, help='Directory of basecalled reads (FAST5)', default=argparse.SUPPRESS)
    parser.add_argument('-t','--threads', type=int, default=1, help='Number of CPU threads for preprocessing')
    parser.add_argument('-m','--model', help='Neural network model')
    parser.add_argument('-w','--weights', help='Neural network weights (if using JSON for --model')
    parser.add_argument('--trimmed_reads', help='Trimmed reads used for assembly, needed if different than raw basecalls (FASTA/FASTQ)', default=argparse.SUPPRESS)
    parser.add_argument('--gpu', type=int, help='GPU to use for neural network acceleration')
    parser.add_argument('--nested_reads', action='store_true', help='Look for FAST5 reads in subfolders')
    parser.add_argument('--window_size', type=int, default=64, help='Draft window size')
    parser.add_argument('--max_time', type=int, default=80, help='Max number of signals per event')

    args = parser.parse_args()

    # load draft sequence (seq also in pileup)
    if args.draft:
        (ref_id, ref_seq) = semapore.util.load_fastx(args.draft, "fasta")[0]

    # load alignment
    pileup = semapore.util.get_pileup_alignment(alignment=args.alignment, reference=args.draft)
    draft_id = pileup.columns[0]
    pileup_reads = list(pileup.columns)[1:]

    # load reads
    if args.nested_reads:
        reads_table = semapore.util.find_nested_reads(args.reads)
        reads = semapore.util.get_reads(pileup_reads, paths=reads_table)
    else:
        reads = semapore.util.get_reads(pileup_reads, dir=args.reads)

    # load trimmed read sequences
    if args.trimmed_reads:
        semapore.util.trim_raw_reads(reads, args.trimmed_reads)

    # load trained model
    if os.path.isdir(args.model):
        # load model from directory i.e. from Model.save()
        model = tf.keras.models.load_model(args.model, compile=False)
    else:
        # otherwise, load architecture from JSON file
        with open(args.model) as json_file:
            json_config = json_file.read()
            model = tf.keras.models.model_from_json(json_config)

        # load trained model weights separately
        if os.path.isdir(args.weights):
            model_file = tf.train.latest_checkpoint(args.weights)
        else:
            model_file = args.weights
        model.load_weights(model_file)

    # batch alignment and reads for inference
    sequence_input, signal_input, signal_input_mask, window_bounds = semapore.util.featurize_inputs(pileup, reads, window_size=args.window_size, max_time=args.max_time, draft_first=True)
    draft_input = sequence_input[:,:,0] 
    sequence_input = sequence_input[:,:,1:]

    if args.gpu:
        device = '/gpu:{}'.format(args.gpu)
    else:
        device = '/gpu:0' if tf.test.is_gpu_available() else 'cpu'
            
    with tf.device(device):

        # convert input to expected tensor shapes
        signal_tensor = tf.RaggedTensor.from_tensor(tf.expand_dims(signal_input, axis=3), ragged_rank=2)
        sequence_tensor = tf.RaggedTensor.from_tensor(sequence_input)
        draft_tensor = tf.RaggedTensor.from_tensor(tf.expand_dims(draft_input, axis=1))

        # prediction on input batches
        logits = model.predict((signal_tensor, sequence_tensor, draft_tensor))

    # viterbi decoding and stitch sequences
    decoded_labels = semapore.network.greedy_decode(logits).numpy()
    decoded_seq = ''.join([''.join(np.take(np.array(['A','C','G','T']), x)) for x in decoded_labels])

    # output polish sequence in FASTA format
    print(semapore.util.fasta_format(draft_id+"_semapore", decoded_seq))

if __name__ == "__main__":
    main()
