import random, numpy as np

def msk2edge(lab):
   msk = np.zeros(lab.shape, dtype=np.bool)
   mskr = lab[1:] != lab[:-1]
   mskc = lab[:,1:] != lab[:,:-1]
   msk[1:] |= mskr; msk[:-1] |= mskr
   msk[:,1:] |= mskc; msk[:,:-1] |= mskc
   return msk

def connect_graph(img):
    pair1 = np.concatenate((img[:-1,:,None], img[1:,:,None]), -1)
    pair2 = np.concatenate((img[:,:-1,None], img[:,1:,None]), -1)
    pair = np.vstack((pair1.reshape(-1,2), pair2.reshape(-1,2)))
    pair = pair[pair[:,0]!=pair[:,1]];pair = pair[pair.min(1)>0]
    idx = np.unique(np.sort(pair), axis=0) if len(pair)>0 else []
    dic = {}
    for i in np.unique(img): dic[i] = []
    for i,j in idx:
        dic[i].append(j)
        dic[j].append(i)
    del dic[0]
    return dic

def node_render(conmap, n=5, rand=10, shuffle=True):
   nodes = list(conmap.keys())
   colors = dict(zip(nodes, [0]*len(nodes)))
   counter = dict(zip(nodes, [0]*len(nodes)))
   if shuffle: random.shuffle(nodes)
   while len(nodes)>0:
     k = nodes.pop(0)
     counter[k] += 1
     hist = [1e4] + [0] * n
     for p in conmap[k]:
         hist[colors[p]] += 1
     if min(hist)==0:
         cand = [i for i in range(n+1) if not hist[i]]
         colors[k] = cand[random.randint(0, len(cand)-1)]
         counter[k] = 0
         continue
     hist[colors[k]] = 1e4
     minc = hist.index(min(hist))
     if counter[k]==rand:
         counter[k] = 0
         minc = random.randint(1,n)
     colors[k] = minc
     for p in conmap[k]:
         if colors[p] == minc:
             nodes.append(p)
   lut = np.zeros(len(colors)+1, dtype=np.uint8)
   for c in colors: lut[c] = colors[c]
   return lut

def rgb_mask(img, lab, edge=None):
   cmap = np.array([(0,0,0),(255,0,0),(0,255,0),
      (0,0,255),(255,255,0),(255,0,255)], dtype=np.uint8)
   idx = connect_graph(lab)
   lut = node_render(idx)
   rgb = cmap[lut][lab]
   img = img.reshape((img.shape+(1,))[:3])
   return np.maximum(img, rgb, out=rgb)
