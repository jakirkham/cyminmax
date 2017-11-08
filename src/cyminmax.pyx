cimport cyminmax

import numpy
cimport numpy

include "version.pxi"


def minmax(arr):
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

    arr = arr.ravel()
    out = numpy.empty((2,), dtype=arr.dtype)

    if arr.dtype.type is numpy.bool:
        cyminmax.cminmax[numpy.uint8_t](
            arr.view(numpy.uint8), out.view(numpy.uint8)
        )
    elif arr.dtype.type is numpy.uint8:
        cyminmax.cminmax[numpy.uint8_t](arr, out)
    elif arr.dtype.type is numpy.uint16:
        cyminmax.cminmax[numpy.uint16_t](arr, out)
    elif arr.dtype.type is numpy.uint32:
        cyminmax.cminmax[numpy.uint32_t](arr, out)
    elif arr.dtype.type is numpy.uint64:
        cyminmax.cminmax[numpy.uint64_t](arr, out)
    elif arr.dtype.type is numpy.int8:
        cyminmax.cminmax[numpy.int8_t](arr, out)
    elif arr.dtype.type is numpy.int16:
        cyminmax.cminmax[numpy.int16_t](arr, out)
    elif arr.dtype.type is numpy.int32:
        cyminmax.cminmax[numpy.int32_t](arr, out)
    elif arr.dtype.type is numpy.int64:
        cyminmax.cminmax[numpy.int64_t](arr, out)
    elif arr.dtype.type is numpy.float32:
        cyminmax.cminmax[numpy.float32_t](arr, out)
    elif arr.dtype.type is numpy.float64:
        cyminmax.cminmax[numpy.float64_t](arr, out)
    else:
        raise TypeError("Unsupported type for `arr` of `%s`." % arr.dtype.name)

    return out
