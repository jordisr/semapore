import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def simulate_fake_signal(num_bases=10, signals_per_base=30, signals_per_base_sd=5, signal_sd=0.1):
    '''
    Generate noisy sample data that looks a little like a segmented nanopore squiggle
    '''
    base_mean = (np.random.random(num_bases)*4).astype(int)
    all_signals = []
    all_segments = []
    signal_start = 0

    for i,m in enumerate(base_mean):
        num_signals = int(norm.rvs(signals_per_base,signals_per_base_sd))
        signals = norm.rvs(m, signal_sd, size=num_signals)
        all_segments.append((signal_start, signal_start+num_signals))
        all_signals.extend(signals)
        signal_start += num_signals

    all_signals = np.array(all_signals)
    all_segments = np.array(all_segments)
    all_sequence = np.take(np.array(['A','C','G','T']), base_mean)
    return all_signals, all_segments, all_sequence
