import sys
import os
import glob
import h5py
import matplotlib.pyplot as plt
import numpy as np

def find_nested_reads(dir):
    # return lookup table of nested read paths
    lookup_table = {}
    for r in glob.glob("{}/*/*.fast5".format(dir)):
        lookup_table[os.path.basename(r)] = r
    return lookup_table

def get_reads(read_ids, dir=None, paths=None, scaling=None):
    """Load basecalled FAST5 files

    Args:
        read_ids (list): read filenames
        base_dir (str): directory to look in for reads
        paths (dict): table of read_id.fast5 => file_path

    Returns:
        dict: reads, nested dict with read ids as keys
    """
    reads = {}
    for r in read_ids:
        fast5_file = "{}.fast5".format(r.split("_")[0])
        if paths:
            fast5_path = paths[fast5_file]
        elif dir:
            fast5_path = os.path.join(dir, fast5_file)
        else:
            sys.exit("Must specify directory or dict of paths")
        read_id, signal, segments, sequence = parse_guppy_fast5(fast5_path, scaling=scaling)
        reads[r] = {'id':read_id, 'signal':signal, 'segments':segments, 'sequence':sequence}
    return reads

def plot_signal_segments(signal, segments, sequence=None, alternating_colors = ['#EE1F60', '#00B0DA'], base_colors={'A':'#EE1F60','C':'#00B0DA', 'G':'#FDB515', 'T':'#00A598'}, label_bases=False, overlap=False, color_by_base=False):

    if (label_bases or color_by_base) and sequence is None:
        sys.exit("Sequence is not specified")

    fig, ax = plt.subplots()
    for i, segment in enumerate(segments):
        segment_start, segment_end = segment

        if overlap and i > 0:
            segment_start -= 1

        if color_by_base:
            segment_color = base_colors.get(sequence[i], 'k')
        else:
            segment_color = alternating_colors[i % len(alternating_colors)]
        ax.plot(np.arange(segment_start, segment_end),signal[segment_start:segment_end], c=segment_color)

    if label_bases:
        xtick_labels = segments[:,0]
        ax.set_xticks(xtick_labels)
        ax.set_xticklabels(sequence)
        ax.set_xlabel("sequence")
    else:
        ax.set_xlabel("time")

    ax.set_ylabel("current")

    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(16)

    fig.set_size_inches(20,5)

    return fig, ax

def parse_guppy_fast5(f, scaling=None):
    """Extract signal, sequence, and segmentation from a basecalled FAST5
    """
    # TODO: support multi-fast5

    hdf = h5py.File(f,'r')

    # basic parameters from run
    read_string = list(hdf['/Raw/Reads'].keys())[0]
    read_id = hdf['/Raw/Reads/'+read_string].attrs['read_id']
    read_start_time = hdf['/Raw/Reads/'+read_string].attrs['start_time']
    read_duration = hdf['/Raw/Reads/'+read_string].attrs['duration']

    # raw events and signals
    raw_signal_path = '/Raw/Reads/'+read_string+'/Signal'
    raw_signal = np.array(hdf[raw_signal_path])
    assert(len(raw_signal) == read_duration)

    # for converting raw signal to current (pA)
    alpha = hdf['UniqueGlobalKey']['channel_id'].attrs['digitisation'] / hdf['UniqueGlobalKey']['channel_id'].attrs['range']
    offset = hdf['UniqueGlobalKey']['channel_id'].attrs['offset']
    sampling_rate = hdf['UniqueGlobalKey']['channel_id'].attrs['sampling_rate']

    # rescale signal (should be same as option used in training)
    if scaling == 'standard':
        # standardize
        signal = (raw_signal - np.mean(raw_signal))/np.std(raw_signal)
    elif scaling == 'picoampere':
        # convert to pA
        signal = (raw_signal+offset)/alpha
    elif scaling == 'median':
        # divide by median
        signal = raw_signal / np.median(raw_signal)
    elif scaling == 'rescale':
        signal = (raw_signal - np.mean(raw_signal))/(np.max(raw_signal) - np.min(raw_signal))
    else:
        signal = raw_signal

    # get segmentation move table from most recent basecalling
    basecall_runs = list(filter(lambda x: x[:8] == "Basecall", hdf['Analyses']))

    with_move_table = []
    for b in basecall_runs:
        if "Move" in hdf['Analyses'][b]["BaseCalled_template"]:
            with_move_table.append((hdf['Analyses'][b].attrs["time_stamp"].decode("utf-8"), b))

    last_move_table = sorted(with_move_table, reverse=True)[0]
    basecall_run = last_move_table[1].split('_')[-1]

    # get basecalled sequence
    guppy_fastq = np.array(hdf['Analyses']['Basecall_1D_{}'.format(basecall_run)]['BaseCalled_template']['Fastq'])
    guppy_fastq = guppy_fastq.item().decode("utf-8")
    guppy_seq = guppy_fastq.split('\n')[1]
    guppy_seq_idx = np.array([{'A':0,'C':1,'G':2,'T':3}[x] for x in guppy_seq])

    # stride from basecaller network
    guppy_stride = hdf['Analyses']['Basecall_1D_{}'.format(basecall_run)]['Summary']['basecall_1d_template'].attrs['block_stride']

    # number of signals to skip at beginning
    guppy_skip_signal = hdf['Analyses']['Segmentation_{}'.format(basecall_run)]['Summary']['segmentation'].attrs['first_sample_template']

    # 1s represent new base moving into pore
    guppy_move_table = np.array(hdf['Analyses']['Basecall_1D_{}'.format(basecall_run)]['BaseCalled_template']['Move'])

    # raw signal size matches move table
    assert (len(signal) - guppy_skip_signal)//guppy_stride == len(guppy_move_table)

    # number of moves corresponds to basecalled sequence
    len(guppy_seq) == np.sum(guppy_move_table)

    guppy_segment_starts = np.where(guppy_move_table == 1)[0]*guppy_stride + guppy_skip_signal
    guppy_segments = np.zeros(shape=(2,len(guppy_segment_starts)), dtype=np.int64)
    guppy_segments[0] = guppy_segment_starts
    guppy_segments[1,:-1] = guppy_segment_starts[1:]
    guppy_segments[1,-1] = len(guppy_move_table)
    guppy_segments = guppy_segments.T

    return read_id, signal, guppy_segments, guppy_seq
