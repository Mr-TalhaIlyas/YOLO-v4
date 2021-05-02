import torch
from torch import nn
import torch.nn.functional as F
import re

class Mish(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        x = x * (torch.tanh(torch.nn.functional.softplus(x)))
        return x


class Upsample(nn.Module):
    def __init__(self):
        super(Upsample, self).__init__()

    def forward(self, x, target_size):
        assert (x.data.dim() == 4)
        _, _, H, W = target_size
        return F.interpolate(x, size=(H, W), mode='nearest')


class Conv_Bn_Activation(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation, bn=True, bias=False):
        super().__init__()
        pad = (kernel_size - 1) // 2

        self.conv = nn.ModuleList()
        if bias:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad))
        else:
            self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, pad, bias=False))
        if bn:
            self.conv.append(nn.BatchNorm2d(out_channels))
        if activation == "mish":
            self.conv.append(Mish())
        elif activation == "relu":
            self.conv.append(nn.ReLU(inplace=True))
        elif activation == "leaky":
            self.conv.append(nn.LeakyReLU(0.1, inplace=True))
        elif activation == "linear":
            pass
        else:
            print("activate error !!! {} {} {}".format(sys._getframe().f_code.co_filename,
                                                       sys._getframe().f_code.co_name, sys._getframe().f_lineno))

    def forward(self, x):
        for l in self.conv:
            x = l(x)
        return x


class ResBlock(nn.Module):
    """
    Sequential residual blocks each of which consists of \
    two convolution layers.
    Args:
        ch (int): number of input and output channels.
        nblocks (int): number of residual blocks.
        shortcut (bool): if True, residual tensor addition is enabled.
    """

    def __init__(self, ch, nblocks=1, shortcut=True):
        super().__init__()
        self.shortcut = shortcut
        self.module_list = nn.ModuleList()
        for i in range(nblocks):
            resblock_one = nn.ModuleList()
            resblock_one.append(Conv_Bn_Activation(ch, ch, 1, 1, 'mish'))
            resblock_one.append(Conv_Bn_Activation(ch, ch, 3, 1, 'mish'))
            self.module_list.append(resblock_one)

    def forward(self, x):
        for module in self.module_list:
            h = x
            for res in module:
                h = res(h)
            x = x + h if self.shortcut else h
        return x


class DownSample1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(3, 32, 3, 1, 'mish')

        self.conv2 = Conv_Bn_Activation(32, 64, 3, 2, 'mish')
        self.conv3 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -2
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')

        self.conv5 = Conv_Bn_Activation(64, 32, 1, 1, 'mish')
        self.conv6 = Conv_Bn_Activation(32, 64, 3, 1, 'mish')
        # [shortcut]
        # from=-3
        # activation = linear

        self.conv7 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # [route]
        # layers = -1, -7
        self.conv8 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # route -2
        x4 = self.conv4(x2)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        # shortcut -3
        x6 = x6 + x4

        x7 = self.conv7(x6)
        # [route]
        # layers = -1, -7
        x7 = torch.cat([x7, x3], dim=1)
        x8 = self.conv8(x7)
        return x8


class DownSample2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(64, 128, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')
        # r -2
        self.conv3 = Conv_Bn_Activation(128, 64, 1, 1, 'mish')

        self.resblock = ResBlock(ch=64, nblocks=2)

        # s -3
        self.conv4 = Conv_Bn_Activation(64, 64, 1, 1, 'mish')
        # r -1 -10
        self.conv5 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample3(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(256, 128, 1, 1, 'mish')

        self.resblock = ResBlock(ch=128, nblocks=8)
        self.conv4 = Conv_Bn_Activation(128, 128, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(256, 512, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(512, 256, 1, 1, 'mish')

        self.resblock = ResBlock(ch=256, nblocks=8)
        self.conv4 = Conv_Bn_Activation(256, 256, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class DownSample5(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(512, 1024, 3, 2, 'mish')
        self.conv2 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'mish')

        self.resblock = ResBlock(ch=512, nblocks=4)
        self.conv4 = Conv_Bn_Activation(512, 512, 1, 1, 'mish')
        self.conv5 = Conv_Bn_Activation(1024, 1024, 1, 1, 'mish')

    def forward(self, input):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x1)

        r = self.resblock(x3)
        x4 = self.conv4(r)

        x4 = torch.cat([x4, x2], dim=1)
        x5 = self.conv5(x4)
        return x5


class Neck(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv3 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        # SPP
        self.maxpool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=5 // 2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=9, stride=1, padding=9 // 2)
        self.maxpool3 = nn.MaxPool2d(kernel_size=13, stride=1, padding=13 // 2)

        # R -1 -3 -5 -6
        # SPP
        self.conv4 = Conv_Bn_Activation(2048, 512, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # UP
        self.upsample1 = Upsample()
        # R 85
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        # R -1 -3
        self.conv9 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv11 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv12 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # UP
        self.upsample2 = Upsample()
        # R 54
        self.conv15 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        # R -1 -3
        self.conv16 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')
        self.conv19 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv20 = Conv_Bn_Activation(256, 128, 1, 1, 'leaky')

    def forward(self, input, downsample4, downsample3):
        x1 = self.conv1(input)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        # SPP
        m1 = self.maxpool1(x3)
        m2 = self.maxpool2(x3)
        m3 = self.maxpool3(x3)
        spp = torch.cat([m3, m2, m1, x3], dim=1)
        # SPP end
        x4 = self.conv4(spp)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        # UP
        up = self.upsample1(x7, downsample4.size())
        # R 85
        x8 = self.conv8(downsample4)
        # R -1 -3
        x8 = torch.cat([x8, up], dim=1)

        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        x11 = self.conv11(x10)
        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)

        # UP
        up = self.upsample2(x14, downsample3.size())
        # R 54
        x15 = self.conv15(downsample3)
        # R -1 -3
        x15 = torch.cat([x15, up], dim=1)

        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        x19 = self.conv19(x18)
        x20 = self.conv20(x19)
        return x20, x13, x6


class Yolov4Head(nn.Module):
    def __init__(self, output_ch):
        super().__init__()
        self.conv1 = Conv_Bn_Activation(128, 256, 3, 1, 'leaky')
        self.conv2 = Conv_Bn_Activation(256, output_ch, 1, 1, 'linear', bn=False, bias=True)
        # self.yolo1 = YoloLayer(anchor_mask=[0, 1, 2], num_classes=80,
        #                        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        #                        num_anchors=9, stride=8)

        # R -4
        self.conv3 = Conv_Bn_Activation(128, 256, 3, 2, 'leaky')

        # R -1 -16
        self.conv4 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv5 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv6 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv7 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv8 = Conv_Bn_Activation(512, 256, 1, 1, 'leaky')
        self.conv9 = Conv_Bn_Activation(256, 512, 3, 1, 'leaky')
        self.conv10 = Conv_Bn_Activation(512, output_ch, 1, 1, 'linear', bn=False, bias=True)
        # self.yolo2 = YoloLayer(anchor_mask=[3, 4, 5], num_classes=80,
        #                        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        #                        num_anchors=9, stride=16)

        # R -4
        self.conv11 = Conv_Bn_Activation(256, 512, 3, 2, 'leaky')

        # R -1 -37
        self.conv12 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv13 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv14 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv15 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv16 = Conv_Bn_Activation(1024, 512, 1, 1, 'leaky')
        self.conv17 = Conv_Bn_Activation(512, 1024, 3, 1, 'leaky')
        self.conv18 = Conv_Bn_Activation(1024, output_ch, 1, 1, 'linear', bn=False, bias=True)
        # self.yolo3 = YoloLayer(anchor_mask=[6, 7, 8], num_classes=80,
        #                        anchors=[12, 16, 19, 36, 40, 28, 36, 75, 76, 55, 72, 146, 142, 110, 192, 243, 459, 401],
        #                        num_anchors=9, stride=32)

    def forward(self, input1, input2, input3):
        x1 = self.conv1(input1)
        x2 = self.conv2(x1)
        # y1 = self.yolo1(x2)

        x3 = self.conv3(input1)
        # R -1 -16
        x3 = torch.cat([x3, input2], dim=1)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)
        x7 = self.conv7(x6)
        x8 = self.conv8(x7)
        x9 = self.conv9(x8)
        x10 = self.conv10(x9)
        # y2 = self.yolo2(x10)

        # R -4
        x11 = self.conv11(x8)
        # R -1 -37
        x11 = torch.cat([x11, input3], dim=1)

        x12 = self.conv12(x11)
        x13 = self.conv13(x12)
        x14 = self.conv14(x13)
        x15 = self.conv15(x14)
        x16 = self.conv16(x15)
        x17 = self.conv17(x16)
        x18 = self.conv18(x17)
        return [x2, x10, x18]
        # y3 = self.yolo3(x18)
        # return [y1, y2, y3]
        # return y3


class Yolov4(nn.Module):
    def __init__(self, yolov4conv137weight=None, n_classes=80):
        super().__init__()

        output_ch = (4 + 1 + n_classes) * 3

        # backbone
        self.down1 = DownSample1()
        self.down2 = DownSample2()
        self.down3 = DownSample3()
        self.down4 = DownSample4()
        self.down5 = DownSample5()
        # neck
        self.neck = Neck()
        # yolov4conv137
        if yolov4conv137weight:
            _model = nn.Sequential(self.down1, self.down2, self.down3, self.down4, self.down5, self.neck)
            pretrained_dict = torch.load(yolov4conv137weight)

            model_dict = _model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k1: v for (k, v), k1 in zip(pretrained_dict.items(), model_dict)}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            _model.load_state_dict(model_dict)
        # head
        self.head = Yolov4Head(output_ch)

    def forward(self, input):
        d1 = self.down1(input)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        x20, x13, x6 = self.neck(d5, d4, d3)

        output = self.head(x20, x13, x6)
        return output
    
def xywh_2_xyminmax(img_w, img_h, box):
    '''
    input_box : (x, y, w, h)
    output_box : (xmin, ymin, xmax, ymax) @ un_normalized
    '''
    xmin = box[0] - (box[2] / 2)
    ymin = box[1] - (box[3] / 2)
    xmax = box[0] + (box[2] / 2)
    ymax = box[1] + (box[3] / 2)
    
    box_minmax = np.array([xmin*img_w, ymin*img_h, xmax*img_w, ymax*img_h]).astype(np.int)
    box_minmax[box_minmax<0] = 0 # to make -ve values zero
    return box_minmax

def draw_boxes(image_in, confidences, nms_box, det_classes, classes, order='yx_minmax', analysis=False):
    '''
    Parameters
    ----------
    image : RGB image original shape will be resized
    confidences : confidence scores array, shape (None,)
    nms_box : all the b_box coordinates array after NMS, shape (None, 4) => order [y_min, x_min, y_max, x_max]
    det_classes : shape (None,), names  of classes detected
    classes : all classes names in dataset
    '''
    img_h = 416
    img_w = 416
    # rescale and resize image
    image = cv2.resize(image_in, (img_w, img_h))/255
    boxes = np.empty((nms_box.shape))
    if order == 'yx_minmax': # pred
        # form [y_min, x_min, y_max, x_max]  to [x_min, y_min, x_max, y_max]
        # and also making them absolute from relative by mult. wiht img dim.
        boxes[:,1] = nms_box[:,0] * img_h
        boxes[:,0] = nms_box[:,1] * img_w
        boxes[:,3] = nms_box[:,2] * img_h 
        boxes[:,2] = nms_box[:,3] * img_w 
    elif order == 'xy_minmax': # gt
        boxes[:,0] = nms_box[:,0] #* img_w
        boxes[:,1] = nms_box[:,1] #* img_h
        boxes[:,2] = nms_box[:,2] #* img_w 
        boxes[:,3] = nms_box[:,3] #* img_h 
    elif order == 'xy_wh': # yolo foramt
        boxes[:,0] = (nms_box[:,0] - (nms_box[:,2] / 2)) * img_w
        boxes[:,1] = (nms_box[:,1] - (nms_box[:,3] / 2)) * img_h
        boxes[:,2] = (nms_box[:,0] + (nms_box[:,2] / 2)) * img_w 
        boxes[:,3] = (nms_box[:,1] + (nms_box[:,3] / 2)) * img_h 
    
    boxes = (boxes).astype(np.uint16)
    i = 1

    colors =  sns.color_palette("Set2") + sns.color_palette("bright")
    [colors.extend(colors) for i in range(3)]
    bb_line_tinkness = 2
    for result in zip(confidences, boxes, det_classes, colors):
        conf = float(result[0])
        facebox = result[1].astype(np.int16)
        #print(facebox)
        name = result[2]
        color = colors[classes.index(name)]#result[3]
        if analysis and order == 'yx_minmax': # pred
            color = (1., 0., 0.) # red  
            bb_line_tinkness = 4
        if analysis and order == 'xy_minmax': # gt
            color = (0., 1., 0.)  # green 
            bb_line_tinkness = 4
        cv2.rectangle(image, (facebox[0], facebox[1]),
                     (facebox[2], facebox[3]), color, bb_line_tinkness)#255, 0, 0
        label = '{0}: {1:0.3f}'.format(name.strip(), conf)
        label_size, base_line = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_DUPLEX   , 0.7, 1)
        
        if not analysis:
            cv2.rectangle(image, (facebox[0], facebox[1] - label_size[1]),    # top left cornor
                         (facebox[0] + label_size[0], facebox[1] + base_line),# bottom right cornor
                         color, cv2.FILLED)
        
            op = cv2.putText(image, label, (facebox[0], facebox[1]),
                       cv2.FONT_HERSHEY_DUPLEX   , 0.7, (0, 0, 0)) 
        i = i+1
    return image#, boxes, det_classes, np.round(confidences, 3)
#%%
import seaborn as sns
import sys, os, glob, random, cv2
from PIL import Image
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
import matplotlib as mpl
from tool.utils import *
mpl.rcParams['figure.dpi'] = 300

img_paths = glob.glob('/home/user01/data_ssd/Talha/yolo/cells_v4/test/*.png') + \
            glob.glob('/home/user01/data_ssd/Talha/yolo/cells_v4/test/*.jpg')
imgfile = random.choice(img_paths)
plot = False
n_classes = 3

weightfile = '/home/user01/data_ssd/Talha/yolo/cells_v4/checkpoints/Yolov4_epoch35.pth'
namesfile = '/home/user01/data_ssd/Talha/yolo/cells_v4/test/_classes.txt'
op_dir = '/home/user01/data_ssd/Talha/yolo/yolo_v4_eval/'

filelist = [ f for f in os.listdir(op_dir)]# if f.endswith(".png") ]
for f in tqdm(filelist, desc = 'Deleting old files op_dir'):
    os.remove(os.path.join(op_dir, f))
    
model = Yolov4(n_classes=n_classes)

pretrained_dict = torch.load(weightfile, map_location=torch.device('cuda'))
model.load_state_dict(pretrained_dict)

use_cuda = 1
if use_cuda:
    model.cuda()


for i in trange(len(img_paths), desc='Writing files for Eval'):
    imgfile = img_paths[i]
    img = Image.open(imgfile).convert('RGB')
    
    '''
    The output format of boxes is 
    <x><y><w><h><conf1><conf2><class_id>
    ignore conf1 use **conf2** as confidence
    '''
    sized = img.resize((608, 608))
    class_names = load_class_names(namesfile)
    boxes = do_detect(model, sized, 0.5, n_classes,0.4, use_cuda)
    
    if plot:
        plot_boxes(img, boxes, 'predictions2.jpg', class_names)
    
    boxes = np.asarray(boxes).astype(np.float)
    name = os.path.basename(imgfile)[:-4]
    
    coords_xywh = np.zeros((boxes.shape[0], 4)) # droping 5th value
    confd = np.zeros((boxes.shape[0], 1))
    class_ids = np.zeros((boxes.shape[0], 1))
    # assign
    coords_xywh = boxes[:,0:4] # coords
    confd = boxes[:,5] # confidence
    class_ids = boxes[:,6] # class id
    
    coords_xyminmax = []
    det_classes = []
    for i in range(boxes.shape[0]):
        coords_xyminmax.append(xywh_2_xyminmax(img.size[0], img.size[1], coords_xywh[i]))
        det_classes.append(class_names[int(class_ids[i])])
    
    all_bounding_boxnind = []
    for i in range(boxes.shape[0]):
        
        bounding_box = [0.0] * 6
        
        bounding_box[0] = det_classes[i]
        bounding_box[1] = confd[i]
        bounding_box[2] = coords_xyminmax[i][0]
        bounding_box[3] = coords_xyminmax[i][1]
        bounding_box[4] = coords_xyminmax[i][2]
        bounding_box[5] = coords_xyminmax[i][3]
        
        bounding_box = str(bounding_box)[1:-1]# remove square brackets
        bounding_box = bounding_box.replace("'",'')# removing inverted commas around class name
        bounding_box = "".join(bounding_box.split())# remove spaces in between **here dont give space inbetween the inverted commas "".
        all_bounding_boxnind.append(bounding_box)
    
    all_bounding_boxnind = ' '.join(map(str, all_bounding_boxnind))# convert list to string
    all_bounding_boxnind=list(all_bounding_boxnind.split(' ')) # convert strin to list
    # replacing commas with spaces
    for i in range(len(all_bounding_boxnind)):
        all_bounding_boxnind[i] = all_bounding_boxnind[i].replace(',',' ')
    for i in range(len(all_bounding_boxnind)):
    # check if file exiscts else make new
        with open(op_dir +'{}.txt'.format(name), "a+") as file_object:
            # Move read cursor to the start of file.
            file_object.seek(0)
            # If file is not empty then append '\n'
            data = file_object.read(100)
            if len(data) > 0 :
                file_object.write("\n")
            # Append text at the end of file
            file_object.write(all_bounding_boxnind[i])
#%%
img  = cv2.imread(imgfile)
t = np.asarray(coords_xyminmax)
op = draw_boxes(img, confd, t, det_classes, class_names, order='xy_minmax', analysis=False)
plt.imshow(op)