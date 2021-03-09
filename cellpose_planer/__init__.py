import planer
from . import cellpose, render

def engine(core, nimg):
    global _asnumpy
    cellpose.np, cellpose.ndimg = core, nimg
    _asnumpy = planer.core(core)
    cellpose.load('cyto')
    print('\nuser switch engine:', core.__name__)

def asnumpy(arr): return _asnumpy.cpu(arr)

try:
    import cupy
    from cupyx.scipy import ndimage as ndimg
    engine(cupy, ndimg)
    print('using cupy engine, gpu powered!')
except:
    import numpy as np
    import scipy.ndimage as ndimg
    engine(cupy, ndimg)
    print('using numpy engine, install cupy would be faster.')
