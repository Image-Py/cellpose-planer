# Cellpose-Planer
![logo](https://user-images.githubusercontent.com/24822467/110975224-5135d880-8314-11eb-9356-69c7a3611cde.png)

[Cellpose](https://github.com/MouseLand/cellpose) is a generalist algorithm for cellular segmentation, Which written by Carsen Stringer and Marius Pachitariu.

[Planer](https://github.com/Image-Py/planer) is a light-weight CNN framework implemented in pure Numpy-like interface. It can run only with Numpy. Or change different backends. (Cupy accelerated with CUDA, ClPy accelerated with OpenCL).

So Cellpose-Planer is the **cellpose** models on **planer** framework. We generate onnx from torch models, then deduce it to planer model.

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
import cellpose as cellpp
from skimage.data import coins

img = coins()
x = img.astype(np.float32)/255

net = cellpp.load_model('cyto_0')
flow, prob, style = cellpp.get_flow(net, x, [0,0])
msk = cellpp.flow2msk(flow * (5/1.5), None, 1, 20, 100)

rgb = cellpp.rgb_mask(img, msk)

plt.subplot(121).imshow(img)
plt.subplot(122).imshow(rgb)
plt.show()
```
![coins](https://user-images.githubusercontent.com/24822467/110975016-1338b480-8314-11eb-84e4-4152e688d577.png)

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
flow, prob, style = cellpp.get_flow(net, gravel(), [0,0])

# you can also load many nets to got a mean output.
nets = cellpp.load_model(['cyto_0', 'cyto_1', 'cyto_2', 'cyto_3'])
flow, prob, style = cellpp.get_flow(net, gravel(), [0,0])

# size parameter force scale the image to size x size.
flow, prob, style = cellpp.get_flow(net, gravel(), [0,0], size=480)

# we can open multi working thread to process each net.
# only useful when multi nets input. (GPU not recommend)
flow, prob, style = cellpp.get_flow(net, gravel(), [0,0], work=4)
```

## 3. processing large image
```python
# processing large image by tiles.
# sample: scale factor
# size: tile's size
# work: number of threads (GPU not recommend)
flow, prob, style = cellpp.tile_flow(net, x, sample=1, size=768, work=4)
```

## 4. flow to mask
```python
# use the flow and prob to build the label
# gradient threshold: the flow below gradient threshold become watershed.
# area threshold: every watershed's area must be small enough
# volume threshold: every watershed's volume must be big enough.
msk = cellpp.flow2msk(flow * (5/1.5), prob, grad=1, area=20, volume=100)
```

## 5. backend and performance
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
from skimage.data import coins, gravel
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
msk = cellpp.flow2msk(flow * (5/1.5), None, 1, 20, 100)
start = time()
msk = cellpp.flow2msk(flow * (5/1.5), None, 1, 20, 100)
print('\tflow time second time:', time()-start)

show(img, cellpp.asnumpy(msk))
```
here is the timing result on I7 CPU, 2070 GPU.
![gravel](https://user-images.githubusercontent.com/24822467/110979607-ac1dfe80-8319-11eb-97d2-6d97f668ebfa.png)
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