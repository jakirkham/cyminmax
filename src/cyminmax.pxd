cimport cython
cimport cpython
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


cdef inline minmax_1d(arr, out):
    """
    Computes the minimum and maximum of an array in one pass.

    This is an optimized version of computing the minimum and maximum
    by doing both in a single pass.

    Args:
        a(array-like):             array to find min and max of.

    Returns:
        out(numpy.ndarray):        an array with the min and max values.
    """

    cdef numpy.NPY_TYPES arr_dtype_num = <numpy.NPY_TYPES>numpy.PyArray_TYPE(
        arr
    )
    cdef size_t arr_size = numpy.PyArray_SIZE(arr)
    cdef void* arr_data = numpy.PyArray_DATA(arr)

    if arr_dtype_num == numpy.NPY_BOOL:
        cminmax[numpy.npy_bool](
            <numpy.npy_bool*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_UBYTE:
        cminmax[numpy.npy_ubyte](
            <numpy.npy_ubyte*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_USHORT:
        cminmax[numpy.npy_ushort](
            <numpy.npy_ushort*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_UINT:
        cminmax[numpy.npy_uint](
            <numpy.npy_uint*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_ULONG:
        cminmax[numpy.npy_ulong](
            <numpy.npy_ulong*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_ULONGLONG:
        cminmax[numpy.npy_ulonglong](
            <numpy.npy_ulonglong*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_BYTE:
        cminmax[numpy.npy_byte](
            <numpy.npy_byte*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_SHORT:
        cminmax[numpy.npy_short](
            <numpy.npy_short*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_INT:
        cminmax[numpy.npy_int](
            <numpy.npy_int*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_LONG:
        cminmax[numpy.npy_long](
            <numpy.npy_long*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_LONGLONG:
        cminmax[numpy.npy_longlong](
            <numpy.npy_longlong*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_FLOAT:
        cminmax[numpy.npy_float](
            <numpy.npy_float*>arr_data, arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_DOUBLE:
        cminmax[numpy.npy_double](
            <numpy.npy_double*>arr_data, arr_size, out
        )
    else:
        raise TypeError("Unsupported type for `arr` of `%s`." % arr.dtype.name)

    return out
