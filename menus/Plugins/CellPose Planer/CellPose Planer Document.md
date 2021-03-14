# CellPose Planer
**def get_flow(nets, img, cn=[0,0], sample=1, size=512, tile=True, work=1, callback=progress)**

* *nets:* the nets loaded upon.

* *img:* the image to process

* *cn:* the cytoplasm and nucleus channels

* *sample:* if not 1, we scale it. (only avalible when tile==True)

* *size:* when tile==True, this is the tile size, when tile==False, we scale the image to size.

* *tile:* if True, method try to process image in tiles. else resize the image.

* *work:* open multi-thread to process the image. (GPU not recommend)
```python
flowpb, style = cellpp.get_flow(net, coins(), [0,0], work=4)
```

**def flow2msk(flowpb, level=0.5, grad=0.5, area=None, volume=None)**

* *flowpb:* get_flow 's output

* *level:* below level means background, where water can not flow. So level decide the outline.

* *grad:* if the flow gradient is smaller than this value, we set it 0. became a watershed. bigger gradient threshold could suppress the over-segmentation. especially in narrow-long area.

* *area:* at end of the flow process, every watershed should be small enough. (<area), default is 0 (auto).

* *volume:* and in small area, must contian a lot of water. (>volume), default is 0 (auto).

```python
msk = cellpp.flow2msk(flowpb, level=0.5, grad=0.5, area=None, volume=None)
```