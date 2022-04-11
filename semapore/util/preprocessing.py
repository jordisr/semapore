from math import ceil
import sys
import numpy as np

def featurize_inputs_ragged(pileup, reads, window_size=100, max_time=150, drop_remainder=False):
    # returns list of examples
    # each example described by dictionary with the following keys:
    # 'signal_values','signal_row_lengths','signal_col_lengths','sequence_values','sequence_col_lengths', 'draft'
    # flat values for constructing ragged tensors
    # either to be serialized or passed to a generator

    num_columns = pileup.get_num_columns()
    if drop_remainder:
        num_columns = (num_columns // window_size)*window_size

    window_bounds = np.zeros((ceil(num_columns / window_size), 2), dtype=np.int32)
    examples = []

    col = 0
    col_i = 0

    num_signals_processed = 0
    num_over_maxtime = 0

    while col < num_columns:
        this_window_size = min(col+window_size, num_columns)-col
        
        signal_values = []
        signal_row_lengths = []
        signal_col_lengths = []
        sequence_values = []
        sequence_col_lengths = []
        draft = []

        pileup_window = pileup.get_window(col, col+window_size)

        for i in range(this_window_size):
            alignment_column = pileup_window.pileup[i]
            alignment_idx = pileup_window.pos[i]

            draft.append(alignment_column[0])
            sequence_values.append(alignment_column[1:])
            sequence_col_lengths.append(len(alignment_column)-1)
           
            n_signal = 0
            for j, read_id in enumerate(pileup_window.reads):
                if alignment_column[j] > 2: # A,C,G,T,a,c,g,t
                    n_signal += 1
                    # alignment index => basecall sequence index
                    basecall_idx = alignment_idx[j]
                    if alignment_column[j] > 6:
                        # correct position if sequence is reverse mapped
                        basecall_idx = reads[read_id]["segments"].shape[0] - basecall_idx - 1

                    # basecall sequence index => signal event bounds
                    event_bounds = reads[read_id]["segments"][basecall_idx]

                    # signal event bounds => raw signal
                    this_signal = reads[read_id]["signal"][event_bounds[0]:event_bounds[1]]

                    # track truncated events
                    num_signals_processed += 1                    
                    if this_signal.shape[0] > max_time:
                        num_over_maxtime += 1
                        num_signals = max_time
                    else:
                        num_signals = this_signal.shape[0]

                    signal_values.append(this_signal[:num_signals])
                    signal_row_lengths.append(num_signals)
 
                else:
                    this_signal = []
            
            signal_col_lengths.append(n_signal)

        window_bounds[col_i] = np.array([col, col+this_window_size])
        col += this_window_size
        col_i += 1

        examples.append({
            'signal_values':np.concatenate(signal_values).astype(np.int16),
            'signal_row_lengths':np.array(signal_row_lengths).astype(np.int32),
            'signal_col_lengths':np.array(signal_col_lengths).astype(np.int32),
            'sequence_values':np.concatenate(sequence_values).astype(np.int16),
            'sequence_col_lengths':np.array(sequence_col_lengths).astype(np.int32),
            'draft':np.array(draft).astype(np.int16)
            })

    print("{}/{} ({:.2f}%) events truncated to max_time={}".format(num_over_maxtime, num_signals_processed, num_over_maxtime/num_signals_processed*100, max_time), file=sys.stderr)
    return examples, window_bounds
