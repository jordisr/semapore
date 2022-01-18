import numpy as np
from scipy.stats import norm

from .pileup import replace_tuple

def featurize_inputs(pileup, reads=False, window_size=100, max_time=100):
    """
    Generate padded input features from the raw signal and reads-to-assembly alignment

    Inputs:
        pileup       output of get_pileup_alignment()
        reads        output of get_fast5_reads()

    Outputs:
        alignment    [?, window_size, num_reads]
        signal       [?, window_size, num_reads, max_time]
        signal_masks [?, window_size, num_reads, max_time]
    """

    num_columns = pileup.shape[0]
    num_reads = pileup.shape[1]

    # more efficient to specify shapes ahead of time instead of padding
    # using 100 as default upper bound for number of signals/base
    # TODO dynamically resize if it exceeds
    # TODO add in padding for last incomplete batch
    # num_reads is total number of reads, possibly impractical for large assemblies
    signal_data = np.zeros((num_columns // window_size, window_size, num_reads, max_time), dtype=np.float32)
    signal_mask = np.zeros((num_columns // window_size, window_size, num_reads, max_time), dtype=bool)
    sequence_data = []

    # same padding across all windows, saves hassle of nested RaggedTensor
    max_signals = 0

    col = 0
    col_i = 0
    while col+window_size < num_columns:
        sequence_cols = []

        # should this be done here or automatically from get_pileup_alignment()?
        pileup_window = pileup.iloc[col:col+window_size].applymap(replace_tuple).to_numpy()

        # can we use more vectorization?
        for i in range(window_size):
            alignment_column = np.array([x[1] for x in pileup_window[i]])
            sequence_cols.append(alignment_column)
            alignment_idx = np.array([x[0] for x in pileup_window[i]])

            for j in range(num_reads):
                if alignment_column[j] > 2: # A,C,G,T,a,c,g,t

                    # alignment index => basecall sequence index
                    basecall_idx = alignment_idx[j]

                    # basecall sequence index => signal event bounds
                    # event_bounds = reads[j]["segments"][basecall_idx]

                    # signal event bounds => raw signal
                    # this is a placeholder for querying FAST5 file
                    num_signals = int(norm.rvs(9,2))
                    this_signal = np.random.random(num_signals)

                    signal_data[col_i, i, j, :num_signals] = this_signal[:max_time]
                    signal_mask[col_i, i, j, :num_signals] = True

                    if num_signals > max_time:
                        # TODO log warning
                        print("Warning")
                    elif num_signals > max_signals:
                        max_signals = num_signals

                else:
                    this_signal = []

        col += window_size
        col_i += 1

        sequence_data.append(sequence_cols)

    return np.array(sequence_data), signal_data[:,:,:,:max_signals], signal_mask[:,:,:,:max_signals]
