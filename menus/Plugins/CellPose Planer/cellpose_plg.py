import numpy as np
from sciapp.action import Simple, Free
from ....import cellpose_planer as cellpp

class Download(Free):
    title = 'CellPose Models Download'
    para = {'models':[]}

    def load(self):
        cellpp.search_models()
        has = cellpp.list_models()
        allms = sorted(cellpp.models.keys())
        allms = [(i+' '*50)[:20]+('--','Installed')[i in has] for i in allms]
        self.view = [('chos', 'models', allms, 'All Models')]
        return True


    def run(self, para):
        models = [i[:20].strip() for i in para['models']]
        cellpp.download(models, self.app.info, self.progress)
        self.app.alert('Download models successfully!')

class CountFlow(Simple):
    title = 'Count Body Flow'
    note = ['all']
    para = {'model':'cyto_0', 'cytoplasm':0, 'nucleus':0, 'rainbow':True, 'prob':True, 
        'zoom':1.0, 'size':512, 'tile':True, 'work':1, 'slice':False}
    view = [(list, 'model', cellpp.list_models(), str, 'model', ''),
            (list, 'cytoplasm', [0,1,2,3], int, 'cytoplasm', 'channel'),
            (list, 'nucleus', [0,1,2,3], int, 'nucleus', 'channel'),
            (float, 'zoom', (0.1, 2), 1, 'zoom', 'pix'),
            (list, 'size', [320, 480, 512, 768, 1024], int, 'size', 'pix'), 
            (bool, 'tile', 'run by tiles'),
            (list, 'work', [1,2,3,4], int, 'work', 'thread'),
            (bool, 'rainbow', 'show rainbow flow'),
            (bool, 'prob', 'show prob body'),
            (bool, 'slice', 'slice')]

    def load(self, ips):
        CountFlow.view[0] = (list, 'model', cellpp.list_models(), str, 'model', '')
        CountFlow.view[1] = (list, 'cytoplasm', list(range(ips.channels)), int, 'cytoplasm', 'channel')
        CountFlow.view[2] = (list, 'nucleus', list(range(ips.channels)), int, 'nucleus', 'channel')
        return True

    # def get_flow(nets, img, cn=[0,0], sample=1, size=512, tile=True, work=1, callback=progress):
    def run(self, ips, imgs, para):
        if not para['slice']:  imgs = [ips.img]
        flows, rainbow, prob = [], [], []
        net = cellpp.load_model(para['model'])
            
        for i in range(len(imgs)):
            img = imgs[i].astype(np.float32)/255
            flowpb, style = cellpp.get_flow(net, img, cn=[para['cytoplasm'], para['nucleus']],
                sample=para['zoom'], size=para['size'], tile=para['tile'], work=para['work'],
                callback=lambda a,b,n=len(imgs): self.progress(n*a, i*a+b))
            flowpb = cellpp.asnumpy(flowpb)
            flows.append(flowpb)
            if para['rainbow']: rainbow.append(cellpp.flow2hsv(flowpb))
            if para['prob']: prob.append((flowpb[:,:,2]*255).astype(np.uint8))

        self.app.show_img(flows, ips.title+'-flow') 
        if para['rainbow']: self.app.show_img(rainbow, ips.title+'-rainbow')
        if para['prob']: self.app.show_img(prob, ips.title+'-prob')

class Flow2Msk(Simple):
    title = 'Body Flow To Mask'
    note = ['float', 'preview']
    para = {'level':0.5, 'grad':0.5, 'area':0, 'volume':0, 'slice':False}
    view = [('slide', 'level', (0.2, 0.8), 1, 'level', 'outline'),
            ('slide', 'grad', (0, 5), 1, 'grad', ''),
            (int, 'area', (0, 1024), 0, 'area', ''),
            (int, 'volume', (0, 4096), 0, 'volume', ''),
            (bool, 'slice', 'slice')]

    def load(self, ips): 
        ips.snapshot()
        return True

    def preview(self, ips, para):
        ips.img[:] = ips.snap
        msk = ips.img[:,:,2] < para['level']
        v = np.linalg.norm(ips.img[:,:,:2], axis=-1)
        msk |= v < para['grad']
        ips.img[msk] = 10,-10,-10

    def cancel(self, ips): ips.swap()

    # flow2msk(flowp, level=0.5, grad=0.5, area=None, volume=None)
    def run(self, ips, imgs, para):
        ips.swap()
        if not para['slice']:  imgs = [ips.img]
        labs = []
        for i in range(len(imgs)):
            lab = cellpp.flow2msk(imgs[i], level=para['level'], 
                grad=para['grad'], area=para['area'], volume=para['volume'])
            labs.append(cellpp.asnumpy(lab))
            self.progress(i+i, len(imgs))
        self.app.show_img(labs, ips.title+'-lab')

class FlowRender(Simple):
    title = 'Render Original With Label'
    note = ['8-bit', 'rgb']
    para = {'lab':None, 'rgb':True, 'line':True}
    view = [('img', 'lab', 'label', 'img'),
            (bool, 'rgb', 'rgb mask'),
            (bool, 'line', 'red line')]

    def run(self, ips, imgs, para):
        ips_lab = self.app.get_img(para['lab'])
        labs = [ips_lab.img] if len(imgs)==1 else ips_lab.imgs
        edges, rgbs = [], []
        for i in range(len(labs)):
            if para['rgb']: 
                rgbs.append(cellpp.rgb_mask(imgs[i], labs[i]))
            if para['line']:
                edges.append(cellpp.draw_edge(imgs[i], labs[i]))
            self.progress(i+1, len(labs))
        if para['rgb']: self.app.show_img(rgbs, ips.title+'-rgb')
        if para['line']: self.app.show_img(edges, ips.title+'-edge')

plgs = [CountFlow, Flow2Msk, FlowRender, '-', Download]
