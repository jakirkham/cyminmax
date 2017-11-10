#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import absolute_import

import pytest

import numpy

import cyminmax


def test_minmax_err():
    arr = numpy.array(b"boo")

    with pytest.raises(TypeError):
        cyminmax.minmax(arr)


@pytest.mark.parametrize("seed", [
    0,
    23,
    796512,
])
@pytest.mark.parametrize("shape", [
    None,
    tuple(),
    (1,),
    (1, 1),
    (2,),
    (3,),
    (4,),
    (5,),
    (10),
    (11),
    (10, 11),
])
@pytest.mark.parametrize("dtype", [
    numpy.uint8,
    numpy.uint16,
    numpy.uint32,
    numpy.uint64,
    numpy.int8,
    numpy.int16,
    numpy.int32,
    numpy.int64,
    numpy.float32,
    numpy.float64,
])
def test_minmax(seed, shape, dtype):
    numpy.random.seed(seed)

    arr = numpy.random.randint(0, 256, shape)

    try:
        arr = arr.astype(dtype)
    except AttributeError:
        arr = dtype(arr)

    expected = numpy.array([arr.min(), arr.max()])
    result = cyminmax.minmax(arr)

    assert result.dtype == expected.dtype
    assert result.shape == expected.shape
    assert (result == expected).all()
