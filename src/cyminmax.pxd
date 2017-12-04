cimport cython
cimport numpy


cdef extern from "numpy/npy_math.h":
    bint npy_isnan(double x) nogil


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
cdef inline void cminmax(real* arr_ptr, size_t arr_size, real[:] out) nogil:
    cdef real arr_max
    cdef real arr_min

    arr_min = arr_max = arr_ptr[0]

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


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.nonecheck(False)
@cython.overflowcheck(False)
@cython.wraparound(False)
cdef inline void cnanminmax(real[::1] arr, real[:] out) nogil:
    cdef real* arr_ptr = &arr[0]
    cdef size_t arr_size = arr.shape[0]

    cdef real arr_max
    cdef real arr_min

    arr_min = arr_max = arr_ptr[0]

    cdef real arr_i_0
    cdef real arr_i_1

    cdef numpy.npy_ubyte arr_i_01_isnan

    cdef size_t i_0 = 1
    cdef size_t i_1
    for i_1 in range(2, arr_size, 2):
        arr_i_0 = arr_ptr[i_0]
        arr_i_1 = arr_ptr[i_1]

        if real is numpy.npy_float or real is numpy.npy_double:
            arr_i_01_isnan = (
                <numpy.npy_ubyte>npy_isnan(arr_i_0) +
                2 * <numpy.npy_ubyte>npy_isnan(arr_i_1)
            )

            if arr_i_01_isnan == 3:
                continue
            if arr_i_01_isnan == 0:
                if arr_i_0 < arr_i_1:
                    if arr_i_0 < arr_min:
                        arr_min = arr_i_0
                    if arr_i_1 > arr_max:
                        arr_max = arr_i_1
                else:
                    if arr_i_0 > arr_max:
                        arr_max = arr_i_0
                    if arr_i_1 < arr_min:
                        arr_min = arr_i_1
            elif arr_i_01_isnan == 1:
                if arr_i_1 < arr_min:
                    arr_min = arr_i_1
                elif arr_i_1 > arr_max:
                    arr_max = arr_i_1
            elif arr_i_01_isnan == 2:
                if arr_i_0 < arr_min:
                    arr_min = arr_i_0
                elif arr_i_0 > arr_max:
                    arr_max = arr_i_0
        else:
            if arr_i_0 < arr_i_1:
                if arr_i_0 < arr_min:
                    arr_min = arr_i_0
                if arr_i_1 > arr_max:
                    arr_max = arr_i_1
            else:
                if arr_i_0 > arr_max:
                    arr_max = arr_i_0
                if arr_i_1 < arr_min:
                    arr_min = arr_i_1

        i_0 += 2

    if i_0 < arr_size:
        arr_i_0 = arr_ptr[i_0]

        if arr_i_0 < arr_min:
            arr_min = arr_i_0
        elif arr_i_0 > arr_max:
            arr_max = arr_i_0

    out[0] = arr_min
    out[1] = arr_max
