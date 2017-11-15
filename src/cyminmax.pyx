cimport cyminmax

import numpy
cimport numpy

include "version.pxi"


numpy.import_array()


def minmax(a):
    """
    Computes the minimum and maximum of an array in one pass.

    This is an optimized version of computing the minimum and maximum
    by doing both in a single pass.

    Args:
        a(array-like):             array to find min and max of.

    Returns:
        out(numpy.ndarray):        an array with the min and max values.
    """

    arr = numpy.asanyarray(a)

    cdef numpy.NPY_TYPES arr_dtype_num = <numpy.NPY_TYPES>numpy.PyArray_TYPE(
        arr
    )
    cdef size_t arr_size = numpy.PyArray_SIZE(arr)

    if not arr_size:
        raise ValueError(
            "zero-size array to reduction operation minmax which has no "
            "identity"
        )

    cdef bint arr_ownsdata = numpy.PyArray_CHKFLAGS(arr, numpy.NPY_OWNDATA)
    if not arr_ownsdata:
        arr = arr.copy()
        arr_ownsdata = True

    cdef numpy.npy_intp out_shape = 2
    out = numpy.PyArray_EMPTY(1, &out_shape, arr_dtype_num, 0)

    if arr_dtype_num == numpy.NPY_BOOL:
        cyminmax.cminmax[numpy.npy_bool](
            <numpy.npy_bool*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_UBYTE:
        cyminmax.cminmax[numpy.npy_ubyte](
            <numpy.npy_ubyte*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_USHORT:
        cyminmax.cminmax[numpy.npy_ushort](
            <numpy.npy_ushort*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_UINT:
        cyminmax.cminmax[numpy.npy_uint](
            <numpy.npy_uint*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_ULONG:
        cyminmax.cminmax[numpy.npy_ulong](
            <numpy.npy_ulong*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_ULONGLONG:
        cyminmax.cminmax[numpy.npy_ulonglong](
            <numpy.npy_ulonglong*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_BYTE:
        cyminmax.cminmax[numpy.npy_byte](
            <numpy.npy_byte*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_SHORT:
        cyminmax.cminmax[numpy.npy_short](
            <numpy.npy_short*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_INT:
        cyminmax.cminmax[numpy.npy_int](
            <numpy.npy_int*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_LONG:
        cyminmax.cminmax[numpy.npy_long](
            <numpy.npy_long*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_LONGLONG:
        cyminmax.cminmax[numpy.npy_longlong](
            <numpy.npy_longlong*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_FLOAT:
        cyminmax.cminmax[numpy.npy_float](
            <numpy.npy_float*>numpy.PyArray_DATA(arr), arr_size, out
        )
    elif arr_dtype_num == numpy.NPY_DOUBLE:
        cyminmax.cminmax[numpy.npy_double](
            <numpy.npy_double*>numpy.PyArray_DATA(arr), arr_size, out
        )
    else:
        raise TypeError("Unsupported type for `arr` of `%s`." % arr.dtype.name)

    return out
