import cython
import numpy as np

cimport cython
from libc.math cimport fabs
cimport numpy as cnp

cnp.import_array()
DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t



@cython.boundscheck(False)
@cython.wraparound(False)
def _calc2sampleKS(cnp.ndarray[DTYPE_t, ndim = 1] samples1_inp, cnp.ndarray[DTYPE_t, ndim = 1] samples2_inp):
    cdef cnp.ndarray[DTYPE_t, ndim = 1] samples1 = samples1_inp.copy()
    cdef cnp.ndarray[DTYPE_t, ndim = 1] samples2 = samples2_inp.copy()
    samples1.sort()
    samples2.sort()
    cdef float diff

    cdef int counter1 = 0
    cdef float y1 = 0.0
    cdef int counter2 = 0
    cdef float y2 = 0.0

    cdef float max_diff = 0.0
    while counter1 < samples1.size and counter2 < samples2.size:
        if samples1[counter1] < samples2[counter2]:
            y1 += 1.0/samples1.size
            counter1 += 1
        elif samples2[counter2] < samples1[counter1]:
            y2 += 1.0/samples2.size
            counter2 += 1
        else:
            y1 += 1.0/samples1.size
            y2 += 1.0/samples2.size
            counter1 += 1
            counter2 += 1

        diff = fabs(y1 - y2)
        if diff > max_diff:
            max_diff = diff

    return max_diff

@cython.boundscheck(False)
@cython.wraparound(False)
def ks_disc_2sample(cnp.ndarray[DTYPE_t, ndim = 1] samples1, cnp.ndarray[DTYPE_t, ndim = 1] samples2, int iters=1000):
    assert samples1.size > 0
    assert samples2.size > 0
    assert iters > 0
    
    cdef int more_than_counter, itr, index1, index2, i 
    # cdef cnp.ndarray[np.uint8_t, ndim = 1] lables
    cdef float org_diff, new_diff
    # Calculate the ks between the samples
    org_diff = _calc2sampleKS(samples1, samples2)
    cdef cnp.ndarray[DTYPE_t, ndim = 1] new_samples1
    cdef cnp.ndarray[DTYPE_t, ndim = 1] new_samples2

    cdef cnp.ndarray[DTYPE_t, ndim = 1] long_samples = np.hstack((samples1,samples2))
    more_than_counter = 0
    for itr in range(iters):
        lables = np.hstack((np.full(samples1.size, False),np.full(samples2.size, True)))
        np.random.shuffle(lables)

        new_samples1 = np.zeros_like(samples1)
        new_samples2 = np.zeros_like(samples2)
        index1 = 0
        index2 = 0
        for i in range(long_samples.size):
            if lables[i]:
                new_samples2[index2] = long_samples[i]
                index2 += 1
            else:
                new_samples1[index1] = long_samples[i]
                index1 += 1

        new_diff = _calc2sampleKS(new_samples1, new_samples2)

        if new_diff > org_diff:
            more_than_counter += 1

    return more_than_counter / float(iters)