# distutils: language = c++
# cython: language_level=3, boundscheck=True, wraparound=True

import numpy as np
import sys
cimport numpy as cnp
from libc.stdint cimport uint16_t, int64_t, uint64_t, uintptr_t
import ctypes