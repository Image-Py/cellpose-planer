import planer
from planer import read_net, resize

import numpy as np
import scipy.ndimage as ndimg

#import cupy as np
#from cupyx.scipy import ndimage as ndimg

pal = planer.core(np)
net = read_net('cellpose')

def cellpose(img):
   return net(np.asarray(img)[None,:,:,:])
      
def flow2msk(flow, prob, grad=1.0, area=150, volume=500):
    shp, dim = flow.shape[:-1], flow.ndim - 1
    l = np.linalg.norm(flow, axis=-1)
    flow /= l.reshape(shp+(1,));flow[l<grad] = 0
    ss = ((slice(None),) * (dim) + ([0,-1],)) * 2
    for i in range(dim):flow[ss[dim-i:-i-2]+(i,)]=0
    sn = np.sign(flow); sn *= 0.5; flow += sn;
    dn = flow.astype(np.int32).reshape(-1, dim)
    strides = np.cumprod(np.array((1,)+shp[::-1]))
    dn = (strides[-2::-1] * dn).sum(axis=-1)
    rst = np.arange(flow.size//dim); rst += dn
    for i in range(10): rst = rst[rst]
    hist = np.bincount(rst, None, len(rst))
    hist = hist.astype(np.uint32).reshape(shp)
    lab, n = ndimg.label(hist, np.ones((3,)*dim))
    volumes = ndimg.sum(hist, lab, np.arange(n+1))
    areas = np.bincount(lab.ravel())
    msk = (areas<area) & (volumes>volume)
    lut = np.zeros(n+1, np.uint32)
    lut[msk] = np.arange(1, msk.sum()+1)
    mask = lut[lab].ravel()[rst].reshape(shp)
    return hist, lut[lab], mask

if __name__ == '__main__':
   import matplotlib.pyplot as plt
   from skimage.segmentation import find_boundaries
   from skimage.data import coins, gravel
   from matplotlib import pyplot as plt
   from time import time

   img = gravel()
   img2c = np.asarray(img[None,:,:])
   img2c = np.concatenate((img2c, img2c))
   img2c = img2c.astype(np.float32)/255
   
   y = cellpose(img2c) # net preheat
   start = time()
   y = cellpose(img2c)
   print('unet detect time:', time()-start)

   water, core, msk = flow2msk(  # flow2msk preheat
        y[0][0,:2].transpose(1,2,0)*(5/1.5), None, 1, 20, 100)
   
   start = time()
   water, core, msk = flow2msk(
        y[0][0,:2].transpose(1,2,0)*(5/1.5), None, 1, 20, 100)
   print('flow2mask time:', time()-start)
   
   hot, msk = pal.cpu(y[0][0]), pal.cpu(msk)

   # ===== plot =====
   plt.subplot(221)
   plt.imshow(gravel(), 'gray')
   plt.title('Image')

   plt.subplot(222)
   plt.imshow(~find_boundaries(msk)*img, 'gray')
   plt.title('Edge')

   plt.subplot(223)
   plt.imshow(hot[1], 'gray')
   plt.title('Flow X')

   plt.subplot(224)
   plt.imshow(hot[0], 'gray')
   plt.title('Flow Y')

   plt.show()
   
