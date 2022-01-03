import argparse

def main():
    parser = argparse.ArgumentParser(description='Neural polishing from signal', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-d','--draft', required=True, help='Draft assembly sequence (FASTA)', default=argparse.SUPPRESS)
    parser.add_argument('-a','--alignment', required=True, help='Reads aligned to draft assembly (BAM)', default=argparse.SUPPRESS)
    parser.add_argument('-r','--reads', required=True, help='Directory of basecalled reads (FAST5)', default=argparse.SUPPRESS)
    parser.add_argument('-t','--threads', type=int, default=1, help='Number of CPU threads for preprocessing')
    parser.add_argument('--device', help='Device to use for neural network acceleration', default=argparse.SUPPRESS)

    args = parser.parse_args()

if __name__ == "__main__":
    main()
