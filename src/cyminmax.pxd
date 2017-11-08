cimport cython
cimport numpy


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


@cython.binding(False)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
cdef inline void cminmax(real[::1] arr, real[::1] out) nogil:
    cdef size_t arr_size = arr.shape[0]

    cdef real* arr_begin = &arr[0]
    cdef real* arr_end = (arr_begin + arr_size)

    cdef real* arr_min = &out[0]
    cdef real* arr_max = (arr_min + 1)

    arr_min[0] = arr_max[0] = arr_begin[0]

    cdef real* arr_pos = arr_begin + 1
    while arr_pos != arr_end:
        if arr_pos[0] < arr_min[0]:
            arr_min[0] = arr_pos[0]
        elif arr_pos[0] > arr_max[0]:
            arr_max[0] = arr_pos[0]
        arr_pos += 1
