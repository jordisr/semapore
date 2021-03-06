import argparse
import glob

from semapore.network import make_training_dir
from semapore.util import find_nested_reads

def main():
    parser = argparse.ArgumentParser(description='Make training data for Semapore', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('pattern', help='Pattern to match directories', default=argparse.SUPPRESS)
    parser.add_argument('-d','--draft', required=True, help='Draft assembly sequence (FASTA)', default=argparse.SUPPRESS)
    parser.add_argument('-a','--alignment', required=True, help='Reads aligned to draft assembly (BAM)', default=argparse.SUPPRESS)
    parser.add_argument('-r','--reads', required=True, help='Directory of basecalled reads (FAST5)', default=argparse.SUPPRESS)
    parser.add_argument('-f','--reference', required=True, help='Reference sequence (FASTA)', default=argparse.SUPPRESS)
    parser.add_argument('-o','--out', default="training", help='Name for output directory')
    parser.add_argument('-t,','--threads', type=int, default=1, help='Number of threads for preprocessing')
    parser.add_argument('--nested', action='store_true', help='Whether to look for reads one level down')
    parser.add_argument('--trimmed_reads', default="", help='Trimmed reads used for assembly, needed if different than raw basecalls (FASTA/FASTQ)')
    parser.add_argument('--window', type=int, default=64, help='Number of pileup alignment columns per input')
    parser.add_argument('--max_time', type=int, default=80, help='Max signal/event size')

    args = parser.parse_args()

    if args.nested:
        reads = find_nested_reads(args.reads)
    else:
        reads = args.reads

    if '*' in args.pattern:
        dirs = glob.glob(args.pattern)
    else:
        with open(args.pattern, 'r') as f:
            dirs = [l.strip('\n') for l in f]
        
    make_training_dir(
                    dirs,
                    draft=args.draft,
                    alignment=args.alignment,
                    reads=reads,
                    reference=args.reference,
                    trimmed_reads=args.trimmed_reads,
                    out=args.out,
                    window=args.window,
                    max_time=args.max_time,
                    threads=args.threads)

if __name__ == "__main__":
    main()
