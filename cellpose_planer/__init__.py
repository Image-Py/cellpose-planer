import numpy as np, planer
from . import cellpose
from .cellpose import load_model, get_flow, tile_flow, flow2msk
from .render import msk2edge, rgb_mask, red_edge, flow2hsv, show
from urllib.request import urlretrieve, urlopen

from glob import glob
import os, os.path as osp
from tqdm import tqdm

def engine(core, nimg):
    global _asnumpy
    cellpose.np, cellpose.ndimg = core, nimg
    _asnumpy = planer.core(core)
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
    engine(np, ndimg)
    print('using numpy engine, install cupy would be faster.')

root = osp.abspath(osp.dirname(__file__))
if not osp.exists(root+'/models'): os.mkdir(root+'/models')

models = {}

def search_models():
    url = 'http://release.imagepy.org/cellpose-planer/README.md'
    lines = urlopen(url).read().decode('utf-8').split('\n')
    global models
    for line in lines:
        if not '[download](' in line: continue
        ss = line.split('|')
        url = ss[-2].split('(')[-1].split(')')[0]
        name = osp.split(url)[1]
        models[ss[1].strip()] = url
    fs = glob(root+'/models/*.npy')
    has = [osp.split(i)[1][:-4] for i in fs]
    for i in sorted(models):
        print(i, ':', 'installed' if i in has else '--')
    
def list_models():
    fs = glob(root+'/models/*.npy')
    print([osp.split(i)[1][:-4] for i in fs])

def progress(a, b, c, bar):
    per = int(min(100.0 * a * b / c, 100))
    bar.update(round(per)-bar.n)

def download(names=['cyto_0','cyto_1','cyto_2','cyto_3']):
    assert len(models)>0, 'please search_models first!'
    if names=='all': names = list(models.keys())
    if isinstance(names, str): names = [names]
    for name in names:
        print('download %s from %s'%(name, models[name]))
        bar = tqdm(total=100)
        urlretrieve(models[name], root+'/models/'+name+'.npy',
            lambda a,b,c,bar=bar: progress(a,b,c,bar))
        urlretrieve(models[name][:-3]+'json',
                    root+'/models/'+name+'.json')

def test():
    from time import time
    from skimage.data import coins
    img = coins()
    x = img.astype(np.float32)/255
    net = load_model('cyto_0')
    start = time()
    flow, prob, style = get_flow(net, x)
    print('\tnet time:', time()-start)
    start = time()
    lab = flow2msk(flow, prob, level=0.2)
    print('\tflow time:', time()-start)
    flow = asnumpy(flow)
    prob = asnumpy(prob)
    lab = asnumpy(lab)
    show(img, flow, prob, lab)
