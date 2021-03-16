cimport numpy
cimport cython
from cython.parallel import prange
from libc.string cimport memset

# https://en.wikipedia.org/wiki/Midpoint_circle_algorithm
cdef void circle(int r, int c, int radius, int* arr, Py_ssize_t step, Py_ssize_t rows, Py_ssize_t cols) nogil except *:
    cdef int err = 0, dx = radius, dy = 0, plus = 1, minus = (radius << 1) - 1
    cdef Py_ssize_t i
    cdef int mask, y11, y12, y21, y22, x11, x12, x21, x22
    cdef int dx_prev=-1, dy_prev=-1
    cdef int *tptr
    while dx >= dy:
        y11 = r - dy; y12 = r + dy; y21 = r - dx; y22 = r + dx
        x11 = c - dx; x12 = c + dx; x21 = c - dy; x22 = c + dy

        if x11 < cols and x12 >= 0:
            x11 = max(0, x11)
            x12 = min(x12, cols-1)
            if <unsigned>y11 < <unsigned>rows:
                tptr = arr + y11 * step
                for i in range(x11, x12+1):
                    tptr[i] += 1
            if <unsigned>y12 < <unsigned>rows and y11 != y12:
                tptr = arr + y12 * step
                for i in range(x11, x12+1):
                    tptr[i] += 1
        if x21 < cols and x22 >= 0:
            if dx_prev != -1 and dx_prev != dx:
                y21 = r - dx_prev
                y22 = r + dx_prev
                x21 = max(0, c - dy_prev)
                x22 = min(c + dy_prev, cols-1)
                if <unsigned>y21 < <unsigned>rows:
                    tptr = arr + y21 * step
                    for i in range(x21, x22+1):
                        tptr[i] += 1
                if <unsigned>y22 < <unsigned>rows:
                    tptr = arr + y22 * step
                    for i in range(x21, x22+1):
                        tptr[i] += 1
        dx_prev = dx
        dy_prev = dy
        dy += 1
        err += plus
        plus += 2
        mask = (err <= 0) - 1
        err -= minus & mask
        dx += mask
        minus -= mask & 2
    if dx_prev > dy_prev:
        y21 = r - dx_prev; y22 = r + dx_prev
        x21 = c - dy_prev; x22 = c + dy_prev
        if x21 < cols and x22 >= 0:
            x21 = max(0, x21)
            x22 = min(x22, cols-1)
            if <unsigned>y21 < <unsigned>rows:
                tptr = arr + y21 * step
                for i in range(x21, x22+1):
                    tptr[i] += 1
            if <unsigned>y22 < <unsigned>rows:
                tptr = arr + y22 * step
                for i in range(x21, x22+1):
                    tptr[i] += 1
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False) 
def recnstrc_by_disk(int[:,:] branches, int[:,:] dist, int[:,::1] recnstrc):
    cdef Py_ssize_t length = branches.shape[0]
    cdef Py_ssize_t i
    cdef int r, c
    cdef int rows = recnstrc.shape[0]
    cdef int cols = recnstrc.shape[1]
    memset(&recnstrc[0, 0], 0, rows*cols*sizeof(int))
    for i in range(length):
        r = branches[i, 0]
        c = branches[i, 1]
        circle(r, c, dist[r, c], &recnstrc[0,0], cols, rows, cols)

cdef int _get_weight(int *ptr1, int *ptr2, int length) nogil:
    cdef int i, res=0
    for i in prange(length):
        res += (ptr1[i] > 0) ^ ((ptr1[i] - ptr2[i]) > 0)
    return res

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False) 
cpdef int get_weight(int[:,::1] recn, int[:,::1] term):
    cdef int rows = recn.shape[0]
    cdef int cols = recn.shape[1]
    return _get_weight(&recn[0, 0], &term[0,0], rows*cols)