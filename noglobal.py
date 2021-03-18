# License:
#   I hereby state this snippet is below "threshold of originality" where applicable (public domain).
#
# Otherwise, since initially posted on Stackoverflow, use as:
#   CC-BY-SA 3.0 skyking, Glenn Maynard, Axel Huebl
#   http://stackoverflow.com/a/31047259/2719194
#   http://stackoverflow.com/a/4858123/2719194

import types

def imports():
    for name, val in globals().items():
        # module imports
        if isinstance(val, types.ModuleType):
            yield name, val
        # functions / callables
        if hasattr(val, '__call__'):
            yield name, val

noglobal = lambda fn: types.FunctionType(fn.__code__, dict(imports()))

# usage example
import numpy as np
import matplotlib.pyplot as plt
import h5py

a = 1

@noglobal
def f(b):
    h5py.is_hdf5("a.tmp")
    # only np. shall be known, not numpy.
    np.arange(4)
    #numpy.arange(4)
    # this var access shall break when called
    print(a)
    print(b)

f(7)
print(a)
