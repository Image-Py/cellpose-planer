import sys; sys.path.append('../')

from cellpose_planer import engine, asnumpy, cellpose, render
import matplotlib.pyplot as plt
from time import time
import numpy as np

from skimage.data import coins, gravel

def show(img, msk):
    rgb = render.rgb_mask(img, msk)
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(rgb)
    plt.show()
    
def np_backend_test():
    print('numpy backend test:')
    import numpy as np
    import scipy.ndimage as ndimg
    engine(np, ndimg)

    img = gravel()
    x = img.astype(np.float32)/255
    
    start = time()
    flow, prob, style = cellpose.get_flow(x, [0,0], 480)
    print('\tnet time:', time()-start)

    start = time()
    msk = cellpose.flow2msk(flow * (5/1.5), None, 1, 20, 100)
    print('\tflow time:', time()-start)

    show(img, msk)

def cp_backend_test():
    print('cupy backend test:')
    import cupy as cp
    import cupyx.scipy.ndimage as cpimg
    engine(cp, cpimg)

    img = gravel()
    x = img.astype(np.float32)/255
    
    start = time()
    flow, prob, style = cellpose.get_flow(x, [0,0])
    print('\tnet time first time (need preheat):', time()-start)
    start = time()
    flow, prob, style = cellpose.get_flow(x, [0,0])
    print('\tnet time second time (faster):', time()-start)
    
    start = time()
    msk = cellpose.flow2msk(flow * (5/1.5), None, 1, 20, 100)
    print('\tflow time first time (need preheat):', time()-start)
    start = time()
    msk = cellpose.flow2msk(flow * (5/1.5), None, 1, 20, 100)
    print('\tflow time second time (faster):', time()-start)

    show(img, asnumpy(msk))

def resize_test():
    print('\nresize test:')
    img = coins()
    x = img.astype(np.float32)/255
    print('if the size is not 64 x n, we need resize it.')
    print('also could be used for speeding up when large image')
    flow, prob, style = cellpose.get_flow(x, [0,0], size=480)
    msk = cellpose.flow2msk(flow * (5/1.5), None, 1, 20, 100)
    show(img, asnumpy(msk))

def large_tile_test():
    print('\nlarge tile test:')
    img = np.tile(np.tile(coins(), 10).T, 10).T
    x = img.astype(np.float32)/255
    print('if the image is too large, we need tile.')
    print('sample: resample the image if it is not 1')
    print('size: tile size, should be 64x, 512/768/1024 recommend')
    print('work: multi thread, useful for multi-core cpu (gpu insensitive)')
    flow, prob, style = cellpose.tile_flow(x, sample=1, size=768, work=1)
    msk = cellpose.flow2msk(flow * (5/1.5), None, 1, 20, 100)
    show(img, asnumpy(msk))
    
if __name__ == '__main__':
    np_backend_test()
    # cp_backend_test()
    resize_test()
    large_tile_test()
