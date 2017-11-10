cimport cyminmax

import numpy
cimport numpy

include "version.pxi"


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

    if not arr.size:
        raise ValueError(
            "zero-size array to reduction operation minmax which has no "
            "identity"
        )

    arr = arr.ravel(order="K")
    out = numpy.empty((2,), dtype=arr.dtype)

    cdef numpy.NPY_TYPES arr_dtype_num = arr.dtype.num
    if arr_dtype_num == numpy.NPY_BOOL:
        cyminmax.cminmax[numpy.npy_bool](arr, out)
    elif arr_dtype_num == numpy.NPY_UBYTE:
        cyminmax.cminmax[numpy.npy_ubyte](arr, out)
    elif arr_dtype_num == numpy.NPY_USHORT:
        cyminmax.cminmax[numpy.npy_ushort](arr, out)
    elif arr_dtype_num == numpy.NPY_UINT:
        cyminmax.cminmax[numpy.npy_uint](arr, out)
    elif arr_dtype_num == numpy.NPY_ULONG:
        cyminmax.cminmax[numpy.npy_ulong](arr, out)
    elif arr_dtype_num == numpy.NPY_ULONGLONG:
        cyminmax.cminmax[numpy.npy_ulonglong](arr, out)
    elif arr_dtype_num == numpy.NPY_BYTE:
        cyminmax.cminmax[numpy.npy_byte](arr, out)
    elif arr_dtype_num == numpy.NPY_SHORT:
        cyminmax.cminmax[numpy.npy_short](arr, out)
    elif arr_dtype_num == numpy.NPY_INT:
        cyminmax.cminmax[numpy.npy_int](arr, out)
    elif arr_dtype_num == numpy.NPY_LONG:
        cyminmax.cminmax[numpy.npy_long](arr, out)
    elif arr_dtype_num == numpy.NPY_LONGLONG:
        cyminmax.cminmax[numpy.npy_longlong](arr, out)
    elif arr_dtype_num == numpy.NPY_FLOAT:
        cyminmax.cminmax[numpy.npy_float](arr, out)
    elif arr_dtype_num == numpy.NPY_DOUBLE:
        cyminmax.cminmax[numpy.npy_double](arr, out)
    else:
        raise TypeError("Unsupported type for `arr` of `%s`." % arr.dtype.name)

    return out
