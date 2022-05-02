import sys
import os
import glob
import h5py
from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np

from .seq import load_fastx, get_sequence_offset

def find_nested_reads(dir):
    # return lookup table of read paths one and two levels deep
    lookup_table = {}
    for r in glob.glob("{}/*.fast5".format(dir))+glob.glob("{}/*/*.fast5".format(dir)):
        lookup_table[os.path.basename(r)] = r
    return lookup_table

class ReadIndex:
    """Cached index of FAST5 reads
        - use sequencing_summary.txt to set mapping of read_id <=> filename
        - build table of filenames and full paths by looking in workspace/
        - LRU cache to load FAST5 reads into memory as needed
        - behaves like a dict mapping read IDs to basecalls
        - can define masked reads covering subsequence of existing read
    """

    def __init__(self, sequencing_summary, maxsize=None, signal_scaling=None):
        self.id2file = {}
        self.file2id = {}
        with open(sequencing_summary, 'r') as f:
            for line in f:
                fields = line.rstrip('\n').split('\t')
                self.file2id[fields[0]] = fields[1]
                self.id2file[fields[1]] = fields[0]
        self.directory = os.path.join(os.path.split(sequencing_summary)[0], "workspace")
        self.file2path = find_nested_reads(self.directory)
        self.maxsize=maxsize
        self.reads = OrderedDict()
        self.signal_scaling=signal_scaling
        self.masked_reads = {}

    def __getitem__(self, key):
        if key in self.masked_reads:
            read, bounds = self.masked_reads[key]
            return self.get_masked_read(masked_read=key, read=self.get_read(read), bounds=bounds)
        else:
            return self.get_read(key)

    def __len__(self):
        return len(self.id2file)
    
    def __contains__(self, key):
        return key in self.id2file

    def get_read(self, read):
        if read in self.reads:
            self.reads.move_to_end(read)
            return self.reads[read]
        result = self.get_read_from_fast5(read)
        self.reads[read] = result
        if self.maxsize is not None and len(self.reads) > self.maxsize:
            self.reads.popitem(last=False)
        return result

    def get_read_from_fast5(self, read):
        fast5_path = self.file2path[self.id2file[read]]
        read_id, signal, segments, sequence = parse_guppy_fast5(fast5_path, scaling=self.signal_scaling)
        return {'id':read_id, 'signal':signal, 'segments':segments, 'sequence':sequence}

    def add_masked_read(self, read, bounds, masked_read=None):
        if masked_read is not None:
            self.masked_reads[masked_read] = (read, bounds)
        else:
            self.masked_reads[read] = (read, bounds)

    def get_masked_read(self, masked_read, read, bounds):
        return {'id':masked_read, 
                'signal':read['signal'], 
                'segments':read['segments'][bounds[0]:bounds[1]], 
                'sequence':read['sequence'][bounds[0]:bounds[1]]}

    def mask_from_fastx(self, f):
        # assuming subreads are numbered with underscores e.g. read1_1, read1_2, etc
        trimmed_reads = load_fastx(f)
        read_offsets = {}
        for r in trimmed_reads:
            subread = r[0]
            sub_seq = r[1]
            read = subread.split('_')[0]
            if self.__contains__(read):
                full_seq = self.get_read(read)['sequence']
                read_offset = get_sequence_offset(full_seq=full_seq, sub_seq=sub_seq)
                read_offsets[subread] = read_offset
                if read_offset:
                    self.add_masked_read(masked_read=subread, read=read, bounds=read_offset)

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
    """Extract signal, sequence, and segmentation from a basecalled FAST5 (multi FAST5 not currently supported)
    """

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
