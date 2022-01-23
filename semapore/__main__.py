import argparse
from . import util

def main():
    parser = argparse.ArgumentParser(description='Neural polishing from signal', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d','--draft', required=True, help='Draft assembly sequence (FASTA)', default=argparse.SUPPRESS)
    parser.add_argument('-a','--alignment', required=True, help='Reads aligned to draft assembly (BAM)', default=argparse.SUPPRESS)
    parser.add_argument('-r','--reads', required=True, help='Directory of basecalled reads (FAST5)', default=argparse.SUPPRESS)
    parser.add_argument('-t','--threads', type=int, default=1, help='Number of CPU threads for preprocessing')
    parser.add_argument('-m','--model', help='Neural network model')
    parser.add_argument('--trimmed_reads', help='Trimmed reads used for assembly, needed if different than raw basecalls (FASTA/FASTQ)', default=argparse.SUPPRESS)
    parser.add_argument('--device', help='Device to use for neural network acceleration', default=argparse.SUPPRESS)

    args = parser.parse_args()

    # load draft sequence (seq also in pileup)
    if args.draft:
        (ref_id, ref_seq) = util.load_fastx(args.draft, "fasta")[0]

    # load alignment
    pileup = util.get_pileup_alignment(alignment=args.alignment, reference=args.draft)
    draft_id = pileup.columns[0]
    draft_pileup = pileup[draft_id]
    pileup.drop(draft_id, axis=1, inplace=True) # drop draft sequence row
    pileup_reads = list(pileup.columns)

    # load reads
    reads = semapore.util.get_reads(pileup_reads, base_dir=args.reads)

    # load trimmed read sequences
    if args.trimmed_reads:
        semapore.util.trim_raw_reads(reads, args.trimmed_reads)

    # batch alignment and reads for inference
    sequence_input, signal_input, signal_input_mask = util.featurize_inputs(pileup)

    # TODO: load neural network model

    # TODO: prediction on input batches

    # TODO: viterbi decoding

    # TODO: stitch sequences together

    # TODO: output fasta

if __name__ == "__main__":
    main()
