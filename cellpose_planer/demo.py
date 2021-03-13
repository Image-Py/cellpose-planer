import sys; sys.path.append('../')
import cellpose_planer as cellpp
import matplotlib.pyplot as plt
from time import time, sleep
import numpy as np

from skimage.data import coins

def search_models():
    cellpp.search_models()
    cellpp.download(['cyto_0', 'cyto_1', 'cyto_1', 'cyto_2'])
    
def np_backend_test():
    print('numpy backend test:')
    import numpy as np
    import scipy.ndimage as ndimg
    cellpp.engine(np, ndimg)

    img = coins()
    x = img.astype(np.float32)/255

    net = cellpp.load_model('cyto_0')
    start = time()
    flow, prob, style = cellpp.get_flow(net, x, [0,0])
    print('\tnet time:', time()-start)

    start = time()
    lab = cellpp.flow2msk(flow, prob, out=0.2)
    print('\tflow time:', time()-start)

    cellpp.show(img, flow, prob, lab)

def cp_backend_test():
    print('cupy backend test:')
    import cupy as cp
    import cupyx.scipy.ndimage as cpimg
    cellpp.engine(cp, cpimg)

    img = coins()
    x = img.astype(np.float32)/255

    net = cellpp.load_model('cyto_0')
    start = time()
    flow, prob, style = cellpp.get_flow(net, x, [0,0])
    print('\tnet time first time (need preheat):', time()-start)
    start = time()
    flow, prob, style = cellpp.get_flow(net, x, [0,0])
    print('\tnet time second time (faster):', time()-start)
    
    start = time()
    lab = cellpp.flow2msk(flow, prob, out=0.2)
    print('\tflow time first time (need preheat):', time()-start)
    start = time()
    lab = cellpp.flow2msk(flow, prob, out=0.2)
    print('\tflow time second time (faster):', time()-start)

    flow, prob, lab = [cellpp.asnumpy(i) for i in (flow, prob, lab)]
    cellpp.show(img, flow, prob, lab)

def resize_test():
    print('\nresize test:')
    img = coins()
    x = img.astype(np.float32)/255
    print('if the size is not 64 x n, we need resize it.')
    print('also could be used for speeding up when large image')
    
    net = cellpp.load_model('cyto_0')
    flow, prob, style = cellpp.get_flow(net, x, [0,0], size=480)
    lab = cellpp.flow2msk(flow, prob)

    flow, prob, lab = [cellpp.asnumpy(i) for i in (flow, prob, lab)]
    cellpp.show(img, flow, prob, lab)

def large_tile_test():
    print('\nlarge tile test:')
    img = np.tile(np.tile(coins(), 10).T, 10).T
    img = img[:,:,None][:,:,[0,0,0]]
    print(img.shape)
    x = img.astype(np.float32)/255
    print('if the image is too large, we need tile.')
    print('sample: resample the image if it is not 1')
    print('size: tile size, should be 64x, 512/768/1024 recommend')
    print('work: multi thread, useful for multi-core cpu (gpu insensitive)')
    net = cellpp.load_model('cyto_0')
    flow, prob, style = cellpp.tile_flow(net, x, sample=1, size=768, work=1)
    import matplotlib.pyplot as plt
    lab = cellpp.flow2msk(flow, prob)
    
    flow, prob, lab = [cellpp.asnumpy(i) for i in (flow, prob, lab)]
    cellpp.show(img, flow, prob, lab)
    
if __name__ == '__main__':
    np_backend_test()
    cp_backend_test()
    resize_test()
    large_tile_test()
