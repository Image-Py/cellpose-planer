import sys; sys.path.append('../')
import cellpose_planer as cellpp
import matplotlib.pyplot as plt
from time import time, sleep
import numpy as np

from skimage.data import coins, gravel

def search_models():
    cpp.search_models()
    cpp.download(['cyto_0', 'cyto_1', 'cyto_1', 'cyto_2'])
    
def show(img, msk):
    rgb = cellpp.rgb_mask(img, msk)
    plt.subplot(121).imshow(img)
    plt.subplot(122).imshow(rgb)
    plt.show()
    
def np_backend_test():
    print('numpy backend test:')
    import numpy as np
    import scipy.ndimage as ndimg
    cpp.engine(np, ndimg)

    img = gravel()
    x = img.astype(np.float32)/255

    net = cpp.load_model('cyto_0')
    start = time()
    flow, prob, style = cpp.get_flow(net, x, [0,0], 480)
    print('\tnet time:', time()-start)

    start = time()
    msk = cpp.flow2msk(flow * (5/1.5), None, 1, 20, 100)
    print('\tflow time:', time()-start)

    show(img, msk)

def cp_backend_test():
    print('numpy backend test:')
    import numpy as np
    import scipy.ndimage as ndimg
    cellpp.engine(np, ndimg)

    img = np.tile(np.tile(gravel(), 2).T, 2).T
    x = img.astype(np.float32)/255

    net = cellpp.load_model('cyto_0')
    start = time()
    flow, prob, style = cellpp.get_flow(net, x, [0,0])
    print('\tnet time first time (need preheat):', time()-start)
    start = time()
    flow, prob, style = cellpp.get_flow(net, x, [0,0])
    print('\tnet time second time (faster):', time()-start)
    
    start = time()
    msk = cellpp.flow2msk(flow * (5/1.5), None, 1, 20, 100)
    print('\tflow time first time (need preheat):', time()-start)
    start = time()
    msk = cellpp.flow2msk(flow * (5/1.5), None, 1, 20, 100)
    print('\tflow time second time (faster):', time()-start)

    show(img, cellpp.asnumpy(msk))

def resize_test():
    print('\nresize test:')
    img = coins()
    x = img.astype(np.float32)/255
    print('if the size is not 64 x n, we need resize it.')
    print('also could be used for speeding up when large image')
    
    net = cellpp.load_model('cyto_0')
    flow, prob, style = cellpp.get_flow(net, x, [0,0], size=0)
    msk = cellpp.flow2msk(flow * (5/1.5), None, 1, 20, 100)
    show(img, cellpp.asnumpy(msk))

def large_tile_test():
    print('\nlarge tile test:')
    img = np.tile(np.tile(coins(), 10).T, 10).T
    x = img.astype(np.float32)/255
    print('if the image is too large, we need tile.')
    print('sample: resample the image if it is not 1')
    print('size: tile size, should be 64x, 512/768/1024 recommend')
    print('work: multi thread, useful for multi-core cpu (gpu insensitive)')
    net = cellpp.load_model('cyto_0')
    flow, prob, style = cellpp.tile_flow(net, x, sample=1, size=768, work=4)
    msk = cellpp.flow2msk(flow * (5/1.5), None, 1, 20, 100)
    show(img, cellpp.asnumpy(msk))
    
if __name__ == '__main__':
    #np_backend_test()
    cp_backend_test()
    #resize_test()
    #large_tile_test()


