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


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
cdef inline void cminmax(real[::1] arr, real[:] out) nogil:
    cdef real* arr_ptr = &arr[0]
    cdef size_t arr_size = arr.shape[0]

    cdef real arr_max
    cdef real arr_min

    arr_min = arr_max = arr[0]

    cdef real arr_i

    cdef size_t i
    for i in range(1, arr_size):
        arr_i = arr_ptr[i]

        if arr_i < arr_min:
            arr_min = arr_i
        elif arr_i > arr_max:
            arr_max = arr_i

    out[0] = arr_min
    out[1] = arr_max
