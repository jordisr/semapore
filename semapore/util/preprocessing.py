from math import ceil, floor
import sys
import numpy as np
from tqdm import tqdm

def featurize_inputs_ragged(pileup, reads, window_size=100, max_time=150, drop_remainder=False):
    # returns list of examples
    # each example described by dictionary with the following keys:
    # 'signal_values','signal_row_lengths','signal_col_lengths','sequence_values','sequence_col_lengths', 'draft'
    # flat values for constructing ragged tensors
    # either to be serialized or passed to a generator

    num_columns = pileup.get_num_columns()
    num_windows = floor(num_columns / window_size) if drop_remainder else ceil(num_columns / window_size)

    window_bounds = np.zeros((num_windows, 2), dtype=np.int32)
    window_bounds[:,0] = np.arange(num_windows) * window_size
    window_bounds[:,1] = window_bounds[:,0] + window_size
    window_bounds[-1,1] = min(window_bounds[-1,1], num_columns)

    examples = []
    num_signals_processed = 0
    num_over_maxtime = 0

    for w in tqdm(window_bounds):
        this_window_size = w[1] - w[0]
        
        signal_values = []
        signal_row_lengths = []
        sequence_values = []
        column_lengths = []
        draft = []

        pileup_window = pileup.get_window(w[0], w[1], min_overlap=0.25)

        for i in range(this_window_size):
            alignment_column = pileup_window.pileup[i]
            alignment_idx = pileup_window.pos[i]

            draft.append(pileup_window.refseq[i])
            sequence_values.append(alignment_column)
            column_lengths.append(len(alignment_column))
           
            for j, read_id in enumerate(pileup_window.reads):
                if alignment_column[j] > 2: # A,C,G,T,a,c,g,t
                    # alignment index => basecall sequence index
                    basecall_idx = alignment_idx[j]
                    if alignment_column[j] > 6:
                        # correct position if sequence is reverse mapped
                        basecall_idx = reads[read_id]["segments"].shape[0] - basecall_idx - 1

                    # basecall sequence index => signal event bounds
                    event_bounds = reads[read_id]["segments"][basecall_idx]

                    # signal event bounds => raw signal
                    this_signal = reads[read_id]["signal"][event_bounds[0]:event_bounds[1]]

                else:
                    this_signal = np.array([])

                # track truncated events
                num_signals_processed += 1                    
                if this_signal.shape[0] > max_time:
                    num_over_maxtime += 1
                    num_signals = max_time
                else:
                    num_signals = this_signal.shape[0]

                signal_values.append(this_signal[:num_signals])
                signal_row_lengths.append(num_signals)
            
        examples.append({
            'signal_values':np.concatenate(signal_values).astype(np.int16),
            'signal_row_lengths':np.array(signal_row_lengths).astype(np.int32),
            'sequence_values':np.concatenate(sequence_values).astype(np.int16),
            'column_lengths':np.array(column_lengths).astype(np.int32),
            'draft':np.array(draft).astype(np.int16)
            })

    print("{}/{} ({:.2f}%) events truncated to max_time={}".format(num_over_maxtime, num_signals_processed, num_over_maxtime/num_signals_processed*100, max_time), file=sys.stderr)
    
    return examples
