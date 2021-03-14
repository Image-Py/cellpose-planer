# Cellpose-Planer
[Cellpose](https://github.com/MouseLand/cellpose) is a generalist algorithm for cellular segmentation, Which written by Carsen Stringer and Marius Pachitariu.

[Planer](https://github.com/Image-Py/planer) is a light-weight CNN framework implemented in pure Numpy-like interface. It can run only with Numpy. Or change different backends. (Cupy accelerated with CUDA, ClPy accelerated with OpenCL).

So Cellpose-Planer is the **cellpose** models on **planer** framework. We generate onnx from torch models, then deduce it to planer model. **but we just use cellpose's models, we rewrite all the pre-after processing and render algorithm, So the result is not same as the official one**

## Features
* cellpose-planer is very light, only depend on [Numpy](https://github.com/numpy/numpy) and Scipy.
* cellpose-planer can be accelerated with [Cupy](https://github.com/cupy/cupy).
* without ui, with out object or class, pure function oriented designed.
* optimize cellpose 's pre-after processing and render algorithm, having a better performance and result.

## Install
**pip install cellpose-planer**

Option: *pip install cupy-cuda101* on envidia gpu, install cuda and cupy would get a large acceleration.

# Usage
```python
import cellpose_planer as cellpp
from skimage.data import coins

img = coins()
x = img.astype(np.float32)/255

net = cellpp.load_model('cyto_0')
flow, prob, style = cellpp.get_flow(net, x)
msk = cellpp.flow2msk(flow, prob)

cellpp.show(img, flow, prob, msk)
```
![demo](https://user-images.githubusercontent.com/24822467/111028247-4d549580-83aa-11eb-9bf4-2cb87332530e.png)

## 1. search and download models
search the models you need, and download them. (just one time)
```python
>>> import cellpose_planer as cellpp
>>> cellpp.search_models()
cyto_0 : --
cyto_1 : --
cyto_2 : --
cyto_3 : --
nuclei_0 : --
nuclei_1 : --
nuclei_2 : --
nuclei_3 : --

>>> cellpp.download(['cyto_0', 'cyto_1', 'cyto_2', 'cyto_3'])
download cyto_0 from http://release.imagepy.org/cellpose-planer/cyto_0.npy
100%|█████████████████████████████████████| 100/100 [00:10<00:00,  2.37it/s]
download cyto_1 from http://release.imagepy.org/cellpose-planer/cyto_1.npy
100%|█████████████████████████████████████| 100/100 [00:10<00:00,  2.37it/s]
download cyto_2 from http://release.imagepy.org/cellpose-planer/cyto_2.npy
100%|█████████████████████████████████████| 100/100 [00:10<00:00,  2.37it/s]
download cyto_3 from http://release.imagepy.org/cellpose-planer/cyto_3.npy
100%|█████████████████████████████████████| 100/100 [00:10<00:00,  2.37it/s]

>>> cellpp.list_models()
['cyto_0', 'cyto_1', 'cyto_2', 'cyto_3']
```

## 2. load models and run on your image
```python
from skimage.data import coins, gravel

# load one net, and input one image, two chanels index
net = cellpp.load_model('cyto_0')
flow, prob, style = cellpp.get_flow(net, coins(), [0,0])

# you can also load many nets to got a mean output.
nets = cellpp.load_model(['cyto_0', 'cyto_1', 'cyto_2', 'cyto_3'])
flow, prob, style = cellpp.get_flow(net, coins(), [0,0])

# size parameter force scale the image to size x size.
flow, prob, style = cellpp.get_flow(net, coins(), [0,0], size=480)

# we can open multi working thread to process each net.
# only useful when multi nets input. (GPU not recommend)
flow, prob, style = cellpp.get_flow(net, coins(), [0,0], work=4)
```

## 3. processing large image
when process a large image, we need tile_flow to processing the image by tiles.
```python
# sample: scale factor, we can zoom image befor processing
# size: tile's size, (512, 768, 1024 recommend)
# work: number of threads (GPU not recommend)
flow, prob, style = cellpp.tile_flow(net, x, sample=1, size=512, work=4)
```

## 4. flow to mask
flow to mask is a water flow process. there are 4 parameters here:
1. level: below level means background, where water can not flow. So level decide the outline.
2. gradient: if the flow gradient is smaller than this value, we set it 0. became a watershed. bigger gradient threshold could suppress the over-segmentation. especially in narrow-long area.
3. area: at end of the flow process, every watershed should be small enough. (<area)
4. volume: and in small area, must contian a lot of water. (>volume)
```python
msk = cellpp.flow2msk(flow, prob, level=0.5, grad=0.5, area=None, volume=None)
```
## 5. render
cellpose-planer implements some render styles.
```python
import cellpose_planer as cellpp
import matplotlib.pyplot as plt

fs = glob('./testimg/*.png')
img = 255-imread(fs[10])
x = img.astype(np.float32)/255

net = cellpp.load_model()

flow, prob, style = cellpp.get_flow(net, x)
msk = cellpp.flow2msk(flow, prob)

flow = cellpp.asnumpy(flow)
prob = cellpp.asnumpy(prob)
msk = cellpp.asnumpy(msk)

# get edge from label msask
edge = cellpp.msk2edge(msk)
# get build flow as hsv 2 rgb
hsv = cellpp.flow2hsv(flow)
# 5 colors render (different in neighborhood)
rgb = cellpp.rgb_mask(img, msk)
# draw edge as red line
line = cellpp.red_edge(img, edge)

plt.subplot(221).imshow(img)
plt.subplot(222).imshow(line)
plt.subplot(223).imshow(hsv)
plt.subplot(224).imshow(rgb)
plt.show()
```
![cell](https://user-images.githubusercontent.com/24822467/111029250-93acf300-83b0-11eb-9e83-41bc0cf045dd.png) 

## 6. backend and performance
Planer can run with numpy or cupy backend, by default, cellpose-planer try to use cupy backend, if failed, use numpy backend. But we can change the backend manually. (if you switch backend, the net loaded befor would be useless, reload them pleanse)
```python
import cellpose-planer as cellpp

# use numpy and scipy as backend
import numpy as np
import scipy.ndimage as ndimg
cellpp.engine(np, ndimg)

# use cupy and cupy.scipy as backend
import cupy as cp
import cupyx.scipy.ndimage as cpimg
cellpp.engine(cp, cpimg)
```

then we do a test on a 1024 x 1024 image.
```python
from skimage.data import gravel
img = np.tile(np.tile(gravel(), 2).T, 2).T
x = img.astype(np.float32)/255

# set backend here ...

net = cellpp.load_model('cyto_0')
# gpu need preheat, so run it befor timing
flow, prob, style = cellpp.get_flow(net, x, [0,0])
start = time()
flow, prob, style = cellpp.get_flow(net, x, [0,0])
print('\tnet time second time:', time()-start)

# gpu need preheat, so run it befor timing
msk = cellpp.flow2msk(flow, prob, 1, 20, 100)
start = time()
msk = cellpp.flow2msk(flow, prob, 1, 20, 100)
print('\tflow time second time:', time()-start)
```
here is the timing result on I7 CPU, 2070 GPU.
```
user switch engine: numpy
    net time second time: 11.590887069702148
    flow time first time: 0.07978630065917969

user switch engine: cupy
	net time second time: 0.013962268829345703
	flow time second time: 0.009973287582397461
```

# Model deducing and releasing
Planer only has forward, so we need train the models in torch. then deduc it in planer.

## deduce from torch
```python
# train in cellpose with torch, and export torch as onnx file.
from planer import onnx2planer
onnx2planer(xxx.onnx)
```
then you would get a json file (graph structure), and a npy file (weights).

## model releasing
if you want to share your model in cellpose-planer, just upload the json and npy file generated upon to any public container, then append a record in the **models list** tabel below, and give a pull request.
*infact, when we call cellpp.search_models, cellpp pull the text below and parse them.*

## models list
| model name | auther | description | url |
| --- | --- | --- | --- |
| cyto_0  | carsen-stringer | [for cell cyto segmentation](http://www.cellpose.org/) | [download](http://release.imagepy.org/cellpose-planer/cyto_0.npy) |
| cyto_1  | carsen-stringer | [for cell cyto segmentation](http://www.cellpose.org/) | [download](http://release.imagepy.org/cellpose-planer/cyto_1.npy) |
| cyto_2  | carsen-stringer | [for cell cyto segmentation](http://www.cellpose.org/) | [download](http://release.imagepy.org/cellpose-planer/cyto_2.npy) |
| cyto_3  | carsen-stringer | [for cell cyto segmentation](http://www.cellpose.org/) | [download](http://release.imagepy.org/cellpose-planer/cyto_3.npy) |
| nuclei_0  | carsen-stringer | [for cell nuclear segmentation](http://www.cellpose.org/) | [download](http://release.imagepy.org/cellpose-planer/nuclei_0.npy) |
| nuclei_1  | carsen-stringer | [for cell nuclear segmentation](http://www.cellpose.org/) | [download](http://release.imagepy.org/cellpose-planer/nuclei_1.npy) |
| nuclei_2  | carsen-stringer | [for cell nuclear segmentation](http://www.cellpose.org/) | [download](http://release.imagepy.org/cellpose-planer/nuclei_2.npy) |
| nuclei_3  | carsen-stringer | [for cell nuclear segmentation](http://www.cellpose.org/) | [download](http://release.imagepy.org/cellpose-planer/nuclei_3.npy) |

 *cellpp.search_models function pull the text below and parse them, welcom to release your models here!*