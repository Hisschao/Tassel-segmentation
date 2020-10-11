from nets.deeplab import Deeplabv3
from PIL import Image
import numpy as np
import random
import copy
import os
from osgeo import gdal,gdal_array


class_colors = [[0,0,0],[255,0,0]]
NCLASSES = 2
HEIGHT = 512
WIDTH = 512
path ="../mydatargb/testing/"

##46 wc0.46+acc 比较好
# ep60 wc0.46+precision ep044-loss0.015-val_loss0.024.h5
# ep039-loss0.014-val_loss0.015.h5 WC + MEANIOU
model = model = Deeplabv3(classes=2,input_shape=(HEIGHT,WIDTH,3))
#ep031-loss0.019-val_loss0.031.h5  wc+pre
#ep183-loss0.018-val_loss0.034.h5
model.load_weights("logs/ep183-loss0.018-val_loss0.034.h5")
imgs = os.listdir(path)

for jpg in imgs:

    img = gdal.Open(path + jpg)
    im_width = img.RasterXSize  # 栅格矩阵的列数
    im_height = img.RasterYSize  # 栅格矩阵的行数
    # im_bands = img.RasterCount  # 波段数
    img_new = img.ReadAsArray(0, 0, im_width, im_height, None, WIDTH, HEIGHT)  # 获取数据
    img_new = np.transpose(img_new, (1, 2, 0))
    img_new = img_new / 255
    img_re = img_new.reshape(-1, HEIGHT, WIDTH, 3)

    pr = model.predict(img_re)[0]

    pr = pr.reshape((int(HEIGHT), int(WIDTH),NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT), int(WIDTH),3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((im_width,im_height))
    outpath ="../deeplab_predict"
    # image = Image.blend(old_img,seg_img,0.3)
    seg_img.save(os.path.join(outpath, "%s_predict.jpg" % jpg[:-4]))

