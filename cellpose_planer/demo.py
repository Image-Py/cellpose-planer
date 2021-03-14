import sys; sys.path.append('../')
import cellpose_planer as cellpp
import numpy as np
from skimage.data import coins

def mini_test():
    print('coins test:')
    img = coins()
    x = img.astype(np.float32)/255
    net = cellpp.load_model('cyto_0')
    flowpb, style = cellpp.get_flow(net, x, size=480)
    lab = cellpp.flow2msk(flowpb, level=0.2)
    flowpb, lab = cellpp.asnumpy(flowpb), cellpp.asnumpy(lab)
    cellpp.show(img, flowpb, lab)

def tile_test():
    print('\nlarge tile test:')
    img = np.tile(np.tile(coins(), 4).T, 4).T
    img = img[:,:,None][:,:,[0,0,0]]
    x = img.astype(np.float32)/255
    net = cellpp.load_model(['cyto_0'])
    flowpb, style = cellpp.get_flow(net, x, sample=1, size=512, tile=True, work=1)
    lab = cellpp.flow2msk(cellpp.asnumpy(flowpb), area=20, volume=50)
    flowpb, lab = cellpp.asnumpy(flowpb), cellpp.asnumpy(lab)
    cellpp.show(img, flowpb, lab)
    
if __name__ == '__main__':
    mini_test()
    tile_test()
    
