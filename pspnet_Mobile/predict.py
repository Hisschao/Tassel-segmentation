from nets.pspnet import mobilenet_pspnet
from PIL import Image
import numpy as np
import random
import copy
import os
from osgeo import gdal,gdal_array

class_colors = [[0,0,0],[255,255,255]]
NCLASSES = 2
HEIGHT = 576*3
WIDTH = 576*3

path ="../mydatargb/testing/"
# ep081-loss0.190-val_loss0.231.h5
model = mobilenet_pspnet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
model.load_weights("logs/ep081-loss0.190-val_loss0.231.h5")
imgs = os.listdir(path)

for jpg in imgs:

    ##修改
    img = gdal.Open(path + jpg)
    im_width = img.RasterXSize  # 栅格矩阵的列数
    im_height = img.RasterYSize  # 栅格矩阵的行数
    im_bands = img.RasterCount  # 波段数
    img_new = img.ReadAsArray(0, 0, im_width, im_height, None, WIDTH, HEIGHT)  # 获取数据
    img_new = np.transpose(img_new, (1, 2, 0))
    img_new = img_new / 255
    img_re = img_new.reshape(-1, HEIGHT, WIDTH, 3)

    pr = model.predict(img_re)[0]

    pr = pr.reshape((int(HEIGHT/4), int(WIDTH/4),NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT/4), int(WIDTH/4),3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ((pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((im_width,im_height))

    outpath ="../unet_predict"
    #image = Image.blend(old_img,seg_img,0.3)
    seg_img.save(os.path.join(outpath, "%s_predict.jpg" % jpg[:-4]))


