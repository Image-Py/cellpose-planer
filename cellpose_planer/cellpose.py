from planer import read_net, resize
import numpy as np
import random, math, itertools
import scipy.ndimage as ndimg

import os.path as osp
root = osp.abspath(osp.dirname(__file__))

try:
   from tqdm import tqdm
   def progress(n, i, bar=[None]):
      if bar[0] is None:
         bar[0] = tqdm()
      bar[0].total = n
      bar[0].update(1)
      if n==i: bar[0] = None
except: progress = print

def load(path):
   global net
   net = read_net(root+'/models/'+path)

def get_flow(img, cn=[0,0], size=0):
   if img.ndim==2: img = img[None,:,:]
   img = np.asarray(img)[None, cn, :, :]
   h, w = img.shape[-2:]
   if size>0: img = resize(img, (size,size))
   y, style = net(img)
   if size>0: y = resize(y, (h, w))
   flow = y[0,:2].transpose(1,2,0)
   return flow, y[0,2], style

def make_slice(l, w, mar):
    r = np.linspace(w//2, l-w//2, math.ceil((l-mar)/(w-mar))).astype(int)
    return [slice(i-w//2, i+w//2) for i in r.tolist()]

def grid_slice(H, W, size, mar):
    a, b = make_slice(H, size, mar), make_slice(W, size, mar)
    return list(itertools.product(a, b))

def tile_flow(img, cn=[0,0], sample=1, size=512, work=1, callback=progress):
   if img.ndim==2: img = np.asarray(img[None,:,:])[cn]
   (_, H, W), k = img.shape, sample; h, w = int(H*k), int(W*k)
   simg = img if sample==1 else resize(img[:,:,:], (h, w))
   dh, dw = max(size-h,0), max(size-w,0)
   if max(dh,dw)>0: simg = np.pad(simg, [(0,0),(0,dh),(0,dw)], 'edge')
   rcs = grid_slice(h+dh, w+dw, size, size//10)
   flow = np.zeros((3, h+dh, w+dw), dtype=simg.dtype)
   count = np.zeros(simg.shape[1:], 'uint8')
   def one(sr, sc, s=[0]):
      flw, prob, _ = get_flow(simg[:,sr,sc], slice(None))
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
    return lut[lab].ravel()[rst].reshape(shp)
    return hist, lut[lab], mask
   
