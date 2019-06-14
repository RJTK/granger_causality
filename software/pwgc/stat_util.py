import scipy.stats as sps
import numpy as np


def benjamini_hochberg(P, alpha=0.05, independent=True):
    """
    alpha is the desired bound on the false discovery rate
    (i.e. proportion of false positives against total positives.

    m is the total number of hypotheses that we test, and P is an
    array of P-values.

    Use independent = False if the p-values are not independent of one
    and other.

    NOTE: P must be a 1-dimensional array of p-values, one for each test.

    Return: The BH rejection threshold.  One should reject all P-values s.t.
    P < th_bh.
    """
    P = np.sort(P)
    m = len(P)

    if independent:
        C = 1
    else:
        C = sum(1. / k for k in range(1, m + 1))

    ell = np.arange(1, m + 1) * alpha / (C * m)
    
    R = np.argmax(np.diff(P < ell))
    return P[R]
