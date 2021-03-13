from planer import read_net, resize
import numpy as np
import scipy.ndimage as ndimg
import random, math, itertools
from tqdm import tqdm

import os.path as osp
root = osp.abspath(osp.dirname(__file__))

def progress(n, i, bar=[None]):
   if bar[0] is None:
      bar[0] = tqdm()
   bar[0].total = n
   bar[0].update(1)
   if n==i: bar[0] = None

def load_model(names='cyto_0'):
   if isinstance(names, str): return read_net(root+'/models/'+names)
   return [read_net(root+'/models/'+i) for i in names]

def get_flow(nets, img, cn=[0,0], size=0, work=1):
   if not isinstance(nets, list): nets = [nets]
   if img.ndim==2: img = img[:,:,None]
   if img.shape[0] != 2: img = img.transpose(2,0,1)
   img = np.asarray(img)[None, cn, :, :]
   h, w = img.shape[-2:]
   if size>0: img = resize(img, (size,size))
   dh, dw = (64-h%64)%64*(size==0), (64-w%64)%64*(size==0)
   if max(dh,dw)>0: img = np.pad(img, [(0,0),(0,0),(0,dh),(0,dw)])
   y = np.zeros((1,3)+img.shape[2:], img.dtype)
   style = np.zeros((1,256), img.dtype)
   def one(net, img):
      i, s = net(img)
      y[:] += i; style[:] += s
   if work>1 and len(nets)>1:
      from concurrent.futures import ThreadPoolExecutor
      pool = ThreadPoolExecutor(max_workers=work, thread_name_prefix="flow")
      for net in nets: pool.submit(one, net, img)
      pool.shutdown(wait=True)
   else:
      for net in nets: one(net, img)
   if len(nets)>0:
      y /= len(nets); style /= len(nets)
   if max(dh, dw)>0: y = y[:,:,:h,:w]
   if size>0: y = resize(y, (h, w))
   flow = y[0,:2].transpose(1,2,0)
   return flow, y[0,2], style

def make_slice(l, w, mar):
    r = np.linspace(w//2, l-w//2, math.ceil((l-mar)/(w-mar))).astype(int)
    return [slice(i-w//2, i+w//2) for i in r.tolist()]

def grid_slice(H, W, size, mar):
    a, b = make_slice(H, size, mar), make_slice(W, size, mar)
    return list(itertools.product(a, b))

def tile_flow(nets, img, cn=[0,0], sample=1, size=512, work=1, callback=progress):
   if not isinstance(nets, list): nets = [nets]
   if img.ndim==2: img = np.asarray(img[None,:,:])[cn]
   else: img = np.asarray(img.transpose(2,0,1))[cn]
   (_, H, W), k = img.shape, sample; h, w = int(H*k), int(W*k)
   simg = img if sample==1 else resize(img[:,:,:], (h, w))
   dh, dw = max(size-h,0), max(size-w,0)
   if max(dh,dw)>0: simg = np.pad(simg, [(0,0),(0,dh),(0,dw)])
   rcs = grid_slice(h+dh, w+dw, size, size//10)
   flow = np.zeros((3, h+dh, w+dw), dtype=simg.dtype)
   count = np.zeros(simg.shape[1:], 'uint8')
   def one(sr, sc, s=[0]):
      flw, prob, _ = get_flow(nets, simg[:,sr,sc], slice(None))
      flow[:2, sr, sc] += flw.transpose(2,0,1)
      flow[2, sr, sc] += prob
      count[sr, sc] += 1
      s[0] += 1
      callback(len(rcs), s[0])
   if work>1:
      from concurrent.futures import ThreadPoolExecutor
      pool = ThreadPoolExecutor(max_workers=work, thread_name_prefix="net")
      for i in range(len(rcs)): pool.submit(one, *rcs[i])
      pool.shutdown(wait=True)
   else:
      for slr, slc in rcs: one(slr, slc)
   flow /= count
   if max(dh, dw)>0: flow = flow[:,:h,:w]
   if sample!=1: flow = resize(flow, (H, W))
   return flow[:2].transpose(1,2,0), flow[2], None

def estimate_volumes(arr, sigma=3):
    msk = arr > 50; 
    idx = np.arange(len(arr), dtype=np.uint32)
    idx, arr = idx[msk], arr[msk]
    for k in np.linspace(5, sigma, 5):
       std = arr.std()
       dif = np.abs(arr - arr.mean())
       msk = dif < std * k
       idx, arr = idx[msk], arr[msk]
    return arr.mean(), arr.std()

def flow2msk(flow, prob, level=0.5, grad=0.5, area=None, volume=None):
    shp, dim = flow.shape[:-1], flow.ndim - 1
    l = np.linalg.norm(flow, axis=-1)
    flow = flow/l.reshape(shp+(1,))
    flow[(prob<-np.log(1/level-1))|(l<grad)] = 0
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
    mean, std = estimate_volumes(volumes, 2)
    if volume is None: volume = max(mean-std*3, 50)
    if area is None: area = volumes // 3
    msk = (areas<area) & (volumes>volume)
    lut = np.zeros(n+1, np.uint32)
    lut[msk] = np.arange(1, msk.sum()+1)
    return lut[lab].ravel()[rst].reshape(shp)
    return hist, lut[lab], mask
   
if __name__ == '__main__':
   a = np.random.rand(100)
   a[3] = 2
   idx = filter(a)
