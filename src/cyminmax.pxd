cimport cython
cimport numpy


ctypedef fused real:
    numpy.uint8_t
    numpy.uint16_t
    numpy.uint32_t
    numpy.uint64_t
    numpy.int8_t
    numpy.int16_t
    numpy.int32_t
    numpy.int64_t
    numpy.float32_t
    numpy.float64_t


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
