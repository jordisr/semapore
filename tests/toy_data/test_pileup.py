import pytest
import os
from pathlib import Path

import numpy as np
import mappy

import semapore

def align_reads(draft_path, reads_path):
    out_path = os.path.splitext(reads_path)[0] + ".bam"
    os.system("minimap2 -x map-ont --cs -a {} {} | samtools view -b | samtools sort > {}".format(str(draft_path), str(reads_path), out_path))
    assert os.path.exists(out_path)
    return out_path

def make_data():
    DATA_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)), "data")
    os.mkdir(DATA_DIR)

    # reference sequence
    np.random.seed(42)
    dna_alphabet = np.array(list("ACGT"))
    ref = np.random.randint(low=0, high=4, size=2000)
    with open(os.path.join(DATA_DIR, "reference.fasta"), 'w') as f:
        print(semapore.util.fasta_format(seq=''.join(np.take(dna_alphabet, ref)), name="reference"), file=f)
    with open(os.path.join(DATA_DIR, "reference_revcomp.fasta"), 'w') as f:
        print(semapore.util.fasta_format(seq=''.join(np.take(dna_alphabet[::-1], ref[::-1])), name="reference"), file=f)

    # draft
    draft = ref[500:1500]
    with open(os.path.join(DATA_DIR, "draft.fasta"), 'w') as f:
        print(semapore.util.fasta_format(seq=''.join(np.take(dna_alphabet, draft)), name="draft"), file=f)

    # reads with no overlap
    no_overlap_reads = [draft[:250], draft[250:500], draft[500:750], draft[750:1000]]
    reads_path = os.path.join(DATA_DIR, "no_overlap_reads.fasta")
    with open(reads_path, 'w') as f:
        for i, s in enumerate(no_overlap_reads):
            print(semapore.util.fasta_format(seq=''.join(np.take(dna_alphabet, s)), name="read{}".format(i)), file=f)
    align_reads(os.path.join(DATA_DIR, "draft.fasta"), reads_path)

    # reads with 200bp overlap
    overlap_reads = [draft[:400], draft[200:600], draft[400:800], draft[600:1000]]
    reads_path = os.path.join(DATA_DIR, "overlap_reads.fasta")
    with open(reads_path, 'w') as f:
        for i, s in enumerate(overlap_reads):
            print(semapore.util.fasta_format(seq=''.join(np.take(dna_alphabet, s)), name="read{}".format(i)), file=f)
    align_reads(os.path.join(DATA_DIR, "draft.fasta"), reads_path)

@pytest.fixture
def data_dir():
    return Path(os.path.join(os.path.dirname(os.path.realpath(__file__)), "data"))

def test_fasta_loading(data_dir):
    reference_path = data_dir / "reference.fasta"
    seq = semapore.util.load_fastx(reference_path)[0][1]
    assert len(seq) == 2000

def test_mappy_alignment_full(data_dir):
    reference_path = data_dir / "reference.fasta"
    seq = semapore.util.load_fastx(reference_path)[0][1]
    aligner = mappy.Aligner(str(reference_path), preset="map-ont")
    hit = next(aligner.map(seq))
    assert hit.q_st == 0
    assert hit.q_en - hit.q_st == 2000

def test_mappy_alignment_partial(data_dir):
    reference_path = data_dir / "reference.fasta"
    seq = semapore.util.load_fastx(reference_path)[0][1][:1999]
    aligner = mappy.Aligner(str(reference_path), preset="map-ont")
    hit = next(aligner.map(seq))
    assert hit.q_st == 0
    assert hit.q_en - hit.q_st == len(seq)

def test_no_overlap(data_dir):
    draft_path = data_dir / "draft.fasta"
    aln_path = data_dir / "no_overlap_reads.bam"
    pileup = semapore.util.get_pileup(alignment=aln_path, reference=draft_path)[0]

    assert len(pileup)==1000
    assert pileup.get_num_columns() == 1000
    assert pileup.get_num_reads() == 4

    assert len(pileup.get_window(0,1000).reads) == 4
 
    target_depth = np.ones(1000, dtype=int)
    np.testing.assert_array_equal(pileup.get_depth(), target_depth)

def test_overlap(data_dir):
    draft_path = data_dir / "draft.fasta"
    aln_path = data_dir / "overlap_reads.bam"
    pileup = semapore.util.get_pileup(alignment=aln_path, reference=draft_path)[0]

    assert len(pileup)==1000
    assert pileup.get_num_columns() == 1000
    assert pileup.get_num_reads() == 4
  
    assert len(pileup.get_window(0,200).reads) == 1
    assert len(pileup.get_window(0,400).reads) == 2

    target_depth = np.ones(1000, dtype=int)
    target_depth[200:800] = 2
    np.testing.assert_array_equal(pileup.get_depth(), target_depth)

def test_feature_labels(data_dir):
    aligner = mappy.Aligner(str(data_dir / "reference.fasta"), preset='map-ont')
    aln_path = data_dir / "overlap_reads.bam"
    draft_path = data_dir / "draft.fasta"
    pileup = semapore.util.get_pileup(alignment=aln_path, reference=draft_path)[0]

    feature1 = {'draft':pileup.get_window(0,64).refseq}
    feature2 = {'draft':pileup.get_window(64,128).refseq}
    feature3 = {'draft':pileup.get_window(128,194).refseq}

    labeled_features = semapore.network.add_labels_from_reference([feature1, feature2, feature3], aligner)

    np.testing.assert_array_equal(labeled_features[0]['draft'] - 3, labeled_features[0]['labels'])
    np.testing.assert_array_equal(labeled_features[1]['draft'] - 3, labeled_features[1]['labels'])
    np.testing.assert_array_equal(labeled_features[2]['draft'] - 3, labeled_features[2]['labels'])

def test_feature_labels(data_dir):
    aligner = mappy.Aligner(str(data_dir / "reference.fasta"), preset='map-ont')
    aln_path = data_dir / "overlap_reads.bam"
    draft_path = data_dir / "draft.fasta"
    pileup = semapore.util.get_pileup(alignment=aln_path, reference=draft_path)[0]

    feature = {'draft':pileup.get_window(0,1000).refseq}
    labeled_features = semapore.network.add_labels_from_reference([feature], aligner)

    assert len(labeled_features) == 1
    np.testing.assert_array_equal(labeled_features[0]['draft'] - 3, labeled_features[0]['labels'])

def test_feature_labels_revcomp(data_dir):
    aligner = mappy.Aligner(str(data_dir / "reference_revcomp.fasta"), preset='map-ont')
    aln_path = data_dir / "overlap_reads.bam"
    draft_path = data_dir / "draft.fasta"
    pileup = semapore.util.get_pileup(alignment=aln_path, reference=draft_path)[0]

    feature = {'draft':pileup.get_window(0,1000).refseq}
    labeled_features = semapore.network.add_labels_from_reference([feature], aligner)

    assert len(labeled_features) == 1
    np.testing.assert_array_equal(labeled_features[0]['draft'] - 3, labeled_features[0]['labels'])

if __name__ == "__main__":
    make_data()