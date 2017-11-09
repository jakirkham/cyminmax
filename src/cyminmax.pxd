cimport cython
cimport numpy


from libcpp.pair cimport pair

cdef extern from "<algorithm>" namespace "std" nogil:
     pair[ForwardIt, ForwardIt] minmax_element[ForwardIt](ForwardIt first,
                                                          ForwardIt last) nogil


ctypedef fused real:
    numpy.npy_bool
    numpy.npy_ubyte
    numpy.npy_ushort
    numpy.npy_uint
    numpy.npy_ulong
    numpy.npy_ulonglong
    numpy.npy_byte
    numpy.npy_short
    numpy.npy_int
    numpy.npy_long
    numpy.npy_longlong
    numpy.npy_float
    numpy.npy_double


ctypedef real* real_ptr


@cython.binding(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
cdef inline void cminmax(real[::1] arr, real[::1] out) nogil:
    cdef size_t arr_size = arr.shape[0]

    cdef real_ptr arr_begin = &arr[0]
    cdef real_ptr arr_end = (arr_begin + arr_size)

    cdef pair[real_ptr, real_ptr] res = minmax_element(arr_begin, arr_end)

    out[0] = res.first[0]
    out[1] = res.second[0]
