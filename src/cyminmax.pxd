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
