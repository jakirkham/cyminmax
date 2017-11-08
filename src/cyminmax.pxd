cimport cython
cimport numpy


ctypedef fused real:
    numpy.npy_bool
    numpy.npy_ubyte
    numpy.npy_ushort
    numpy.npy_uint
    numpy.npy_ulong
    numpy.npy_byte
    numpy.npy_short
    numpy.npy_int
    numpy.npy_long
    numpy.npy_float
    numpy.npy_double


@cython.binding(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
cdef inline void cminmax(real[:] arr, real[:] out) nogil:
    cdef real arr_i = arr[0]
    out[:] = arr_i

    for i in range(1, arr.shape[0]):
        arr_i = arr[i]
        if arr_i < out[0]:
            out[0] = arr_i
        elif arr_i > out[1]:
            out[1] = arr_i
