import numpy as np
from scipy.stats import norm

from .pileup import replace_tuple

def featurize_inputs(pileup, reads, window_size=100, max_time=150, trim_down=False, draft_first=False):
    """Generate padded input features from the raw signal and reads-to-assembly alignment

    Args:
        pileup (DataFrame): output of get_pileup_alignment()
        reads (dict): output of get_fast5_reads()
        window_size (int): number of pileup alignment columns per input
        max_time (int): max signal/event size, larger ones will be truncated
        trim_down (bool): whether to reduce signal array if all events < max_time

    Returns:
        ndarray: alignment
            [?, window_size, num_reads]
        ndarray: signal
            [?, window_size, num_reads, max_time]
        ndarray: signal_masks
            [?, window_size, num_reads, max_time]
        ndarray: window_bounds
            [?, 2]
    """

    num_columns = pileup.shape[0]
    num_reads = pileup.shape[1]
    read_names = list(pileup.columns)

    if draft_first:
        read_names = read_names[1:]
        num_reads -= 1

    # TODO: dynamically resize if it exceeds?
    # TODO: add in padding for last incomplete batch
    # num_reads is total number of reads, possibly impractical for large assemblies
    signal_data = np.zeros((num_columns // window_size, window_size, num_reads, max_time), dtype=np.float32)
    signal_mask = np.zeros((num_columns // window_size, window_size, num_reads, max_time), dtype=bool)
    window_bounds = np.zeros((num_columns // window_size, 2), dtype=np.int32)
    sequence_data = []

    # same padding across all windows, saves hassle of nested RaggedTensor
    max_signals = 0

    col = 0
    col_i = 0

    num_signals_processed = 0
    num_over_maxtime = 0
    while col+window_size < num_columns:
        sequence_cols = []

        # should this be done here or automatically from get_pileup_alignment()?
        pileup_window = pileup.iloc[col:col+window_size].applymap(replace_tuple).to_numpy()

        # can we use more vectorization?
        for i in range(window_size):
            alignment_column = np.array([x[1] for x in pileup_window[i]])
            sequence_cols.append(alignment_column)
            alignment_idx = np.array([x[0] for x in pileup_window[i]])

            for j, read_id in enumerate(read_names):
                if alignment_column[j+draft_first] > 2: # A,C,G,T,a,c,g,t

                    # alignment index => basecall sequence index
                    basecall_idx = alignment_idx[j+draft_first]
                    if alignment_column[j+draft_first] > 6:
                        # correct position if sequence is reverse mapped
                        basecall_idx = reads[read_id]["segments"].shape[0] - basecall_idx - 1

                    # basecall sequence index => signal event bounds
                    event_bounds = reads[read_id]["segments"][basecall_idx]
                    #base_from_read = reads[read_id]["sequence"][basecall_idx]
                    #base_from_pileup = ['A','C','G','T','a','c','g','t'][(alignment_column[j+1]-3)]
                    #print(j, base_from_read, base_from_pileup)

                    # signal event bounds => raw signal
                    this_signal = reads[read_id]["signal"][event_bounds[0]:event_bounds[1]]
                    num_signals = this_signal.shape[0]

                    signal_data[col_i, i, j, :num_signals] = this_signal[:max_time]
                    signal_mask[col_i, i, j, :num_signals] = True

                    num_signals_processed += 1
                    if num_signals > max_time:
                        # track truncated events
                        num_over_maxtime += 1
                    elif num_signals > max_signals:
                        max_signals = num_signals

                else:
                    this_signal = []

        window_bounds[col_i] = np.array([col, col+window_size])
        col += window_size
        col_i += 1

        sequence_data.append(sequence_cols)

    if not trim_down:
        max_signals = max_time

    #print("{}% events truncated (max_time={})".format(format(num_over_maxtime/num_signals_processed*100, ".2f"), max_time))
    return np.array(sequence_data), signal_data[:,:,:,:max_signals], signal_mask[:,:,:,:max_signals], window_bounds
