cimport cyminmax

import numpy
cimport numpy

include "version.pxi"


def minmax(arr, axis=None):
    """
    Computes the minimum and maximum of an array in one pass.

    This is an optimized version of computing the minimum and maximum
    by doing both in a single pass.

    Args:
        arr(array-like):           array to find min and max of.

    Returns:
        out(numpy.ndarray):        an array with the min and max values.
    """

    arr = numpy.asanyarray(arr)

    if not arr.size:
        raise ValueError("zero-size array to reduction operation minmax which has no identity")

    if axis is None:
        axis = tuple(range(arr.ndim))

    try:
        axis = tuple(sorted(axis))
    except TypeError:
        axis = (axis,)

    other_axis = tuple(sorted(set(range(arr.ndim)) - set(axis)))
    arr = arr.transpose(other_axis + axis)

    out_shape = tuple(
        s for i, s in enumerate(arr.shape[:len(other_axis)]) if i not in axis
    )
    out_shape += (2,)
    out = numpy.empty(out_shape, dtype=arr.dtype)

    cdef numpy.NPY_TYPES arr_dtype_num = arr.dtype.num

    arr = arr[None]
    for i in numpy.ndindex(arr.shape[:len(other_axis)]):
        arr_i = arr[i].ravel("K")
        out_i = out[i].ravel("K")

        if arr_dtype_num == numpy.NPY_BOOL:
            cyminmax.cminmax[numpy.npy_bool](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_UBYTE:
            cyminmax.cminmax[numpy.npy_ubyte](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_USHORT:
            cyminmax.cminmax[numpy.npy_ushort](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_UINT:
            cyminmax.cminmax[numpy.npy_uint](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_ULONG:
            cyminmax.cminmax[numpy.npy_ulong](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_ULONGLONG:
            cyminmax.cminmax[numpy.npy_ulonglong](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_BYTE:
            cyminmax.cminmax[numpy.npy_byte](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_SHORT:
            cyminmax.cminmax[numpy.npy_short](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_INT:
            cyminmax.cminmax[numpy.npy_int](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_LONG:
            cyminmax.cminmax[numpy.npy_long](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_LONGLONG:
            cyminmax.cminmax[numpy.npy_longlong](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_FLOAT:
            cyminmax.cminmax[numpy.npy_float](arr_i, out_i)
        elif arr_dtype_num == numpy.NPY_DOUBLE:
            cyminmax.cminmax[numpy.npy_double](arr_i, out_i)
        else:
            raise TypeError(
                "Unsupported type for `arr` of `%s`." % arr.dtype.name
            )

    return out
