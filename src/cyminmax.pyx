cimport cyminmax

cimport cpython
cimport numpy

import numpy

include "version.pxi"


cdef extern from "numpy/arrayobject.h":
    ctypedef enum NPY_ORDER:
        NPY_ANYORDER
        NPY_CORDER
        NPY_FORTRANORDER
        NPY_KEEPORDER


numpy.import_array()


def minmax(a, axis=None, out=None):
    """
    Computes the minimum and maximum of an array in one pass.

    This is an optimized version of computing the minimum and maximum
    by doing both in a single pass.

    Args:
        a(array-like):             array to find min and max of.

    Returns:
        out(numpy.ndarray):        an array with the min and max values.
    """

    cpython.Py_INCREF(a)
    arr = numpy.PyArray_EnsureAnyArray(a)

    cdef size_t arr_size = numpy.PyArray_SIZE(arr)

    if not arr_size:
        raise ValueError(
            "zero-size array to reduction operation minmax which has no "
            "identity"
        )

    if axis is None:
        axis = tuple(range(arr.ndim))

    try:
        len(axis)
    except TypeError:
        axis = (axis,)

    axis = tuple(reversed(sorted(set(axis))))

    if axis == 0:
        raise ValueError("`axis` must contain some values if set")

    if len(axis) < arr.ndim:
        for i, ax in range(len(axis)):
            arr = numpy.PyArray_SwapAxes(arr, axis, arr.ndim - i - 1)
        if len(axis) > 1:
            arr = numpy.PyArray_Reshape(arr, arr.shape[:-len(axis)] + (-1,))
        arr = numpy.PyArray_Copy(arr)
    else:
        axis = None
        arr_flat = numpy.PyArray_Ravel(arr, NPY_KEEPORDER)
        if not numpy.PyArray_CHKFLAGS(arr, numpy.NPY_OWNDATA):
            arr_flat = numpy.PyArray_Copy(arr_flat)
        arr = arr_flat[None]

    if out is None:
        out = numpy.empty(arr.shape[:-1] + (2,), dtype=arr.dtype)
    else:
        if out.shape != (arr.shape[:-1] + (2,)):
            raise ValueError("`out` did not match expected shape")
        if out.dtype != arr.dtype:
            raise ValueError("`out` did not match expected type")

    for i in numpy.ndindex(arr.shape[:-1]):
        cyminmax.minmax_1d(arr[i], out[i])

    if axis is None:
        out = out[0]

    return out
