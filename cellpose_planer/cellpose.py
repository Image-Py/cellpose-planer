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

def count_flow(nets, img, cn=[0,0], size=512, work=1):
   if not isinstance(nets, list): nets = [nets]
   img = np.asarray(img)[None, cn, :, :]
   h, w = img.shape[-2:]
   if not(h==w==size): img = resize(img, (size,size))
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
   if not(h==w==size): y = resize(y, (h, w))
   return y[0].transpose(1,2,0), style

def make_slice(l, w, mar):
    r = np.linspace(w//2, l-w//2, math.ceil((l-mar)/(w-mar))).astype(int)
    return [slice(i-w//2, i+w//2) for i in r.tolist()]

def grid_slice(H, W, size, mar):
    a, b = make_slice(H, size, mar), make_slice(W, size, mar)
    return list(itertools.product(a, b))

def get_flow(nets, img, cn=[0,0], sample=1, size=512, tile=True, work=1, callback=progress):
   if not isinstance(nets, list): nets = [nets]
   if img.ndim==2: img = np.asarray(img[None,:,:])[cn]
   else: img = np.asarray(img.transpose(2,0,1))[cn]
   (_, H, W), k = img.shape, sample;
   h, w = (size, size) if not tile else (max(size,int(H*k)), max(int(W*k), size))
   needresize = ((k!=1 or min(H, W)<size) and tile) or (not(H==W==size) and not tile)
   simg = img if not needresize else resize(img[:,:,:], (h, w))
   rcs = grid_slice(h, w, size, size//10)
   flow = np.zeros((3, h, w), dtype=simg.dtype)
   style = np.zeros((1, 256), dtype=simg.dtype)
   count = np.zeros(simg.shape[1:], 'uint8')
   def one(sr, sc, sz, wk, s=[0]):
      flw_prb, sty = count_flow(nets, simg[:,sr,sc], slice(None), sz, wk)
      flow[:, sr, sc] += flw_prb.transpose(2,0,1)
      style[:] += sty
      count[sr, sc] += 1
      s[0] += 1
      callback(len(rcs), s[0])
   if work>1 and len(rcs)>1:
      from concurrent.futures import ThreadPoolExecutor
      pool = ThreadPoolExecutor(max_workers=work, thread_name_prefix="net")
      for i in range(len(rcs)): pool.submit(one, *rcs[i], size, 1)
      pool.shutdown(wait=True)
   else:
      for slr, slc in rcs: one(slr, slc, size, work)
   flow /= count; style /= len(rcs)
   if needresize: flow = resize(flow, (H, W))
   flow[2]*=-1; np.exp(flow[2], out=flow[2]);
   flow[2]+=1; np.divide(1, flow[2], out=flow[2])
   return flow.transpose(1,2,0), style

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

def flow2msk(flowp, level=0.5, grad=0.5, area=None, volume=None):
    flowp = np.asarray(flowp)
    shp, dim = flowp.shape[:-1], flowp.ndim - 1
    l = np.linalg.norm(flowp[:,:,:2], axis=-1)
    flow = flowp[:,:,:2]/l.reshape(shp+(1,))
    flow[(flowp[:,:,2]<level)|(l<grad)] = 0
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
    if not volume: volume = max(mean-std*3, 50)
    if not area: area = volumes // 3
    msk = (areas<area) & (volumes>volume)
    lut = np.zeros(n+1, np.uint32)
    lut[msk] = np.arange(1, msk.sum()+1)
    return lut[lab].ravel()[rst].reshape(shp)
    return hist, lut[lab], mask
   
if __name__ == '__main__':
   a = np.random.rand(100)
   a[3] = 2
   idx = filter(a)
