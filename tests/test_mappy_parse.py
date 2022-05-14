import numpy as np

import semapore

class TestAlignmentCSParsing:

    def test_same(self):
        """
        AATAGAGTAG
        ||||||||||
        AATAGAGTAG
        """
        r2q, q2r = semapore.network.get_alignment_coord(':10')

        np.testing.assert_array_equal(r2q, q2r)

        np.testing.assert_array_equal(r2q, np.arange(10))

    def test_overhang_ref(self):
        """
        ATAAATAGAGTAG
        ||||||||||
        ---AATAGAGTAG
        """
        r2q, q2r = semapore.network.get_alignment_coord('-ata:10')

        np.testing.assert_array_equal(r2q, np.array([0,0,0,0,1,2,3,4,5,6,7,8,9]))

        np.testing.assert_array_equal(q2r, np.array([3,4,5,6,7,8,9,10,11,12]))

    def test_overhang_query(self):
        """
        AATAGAGTAGATA
        ||||||||||
        AATAGAGTAG---
        """
        r2q, q2r = semapore.network.get_alignment_coord(':10-ata')

        np.testing.assert_array_equal(r2q, np.array([0,1,2,3,4,5,6,7,8,9,9,9,9]))

        np.testing.assert_array_equal(q2r, np.array([0,1,2,3,4,5,6,7,8,9]))

    def test_minimap_example(self):
        """
        Example from https://github.com/lh3/minimap2

        CGATCGATAAATAGAGTAG---GAATAGCA
        ||||||   ||||||||||   |||| |||
        CGATCG---AATAGAGTAGGTCGAATtGCA
        """
        r2q, q2r = semapore.network.get_alignment_coord(':6-ata:10+gtc:4*at:3')

        r2q_expected = np.array([0,1,2,3,4,5,5,5,5,6,7,8,9,10,11,12,13,14,15,19,20,21,22,23,24,25,26])
        q2r_expected = np.array([0,1,2,3,4,5,9,10,11,12,13,14,15,16,17,18,18,18,18,19,20,21,22,23,24,25,26])

        np.testing.assert_array_equal(r2q, r2q_expected)

        np.testing.assert_array_equal(q2r, q2r_expected)
