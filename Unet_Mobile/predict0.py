from nets.unet import mobilenet_unet
from PIL import Image
import numpy as np
import random
import copy
import os
from osgeo import gdal,gdal_array

random.seed(0)
class_colors = [[0,0,0],[255,255,255]]
NCLASSES = 2
HEIGHT = 416
WIDTH = 416
#测试数据集
path ="./mydatargb/testing/"

model = mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
model.load_weights("logs/ep050-loss0.230-val_loss0.159.h5")

imgs = os.listdir(path)

for jpg in imgs:
    img = Image.open(path+jpg)
    old_img = copy.deepcopy(img)
    orininal_h = np.array(img).shape[0]
    orininal_w = np.array(img).shape[1]

    img = img.resize((WIDTH,HEIGHT))
    img = np.array(img)
    img = img/255
    img = img.reshape(-1,HEIGHT,WIDTH,3)
    pr = model.predict(img)[0]

    pr = pr.reshape((int(HEIGHT/2), int(WIDTH/2),NCLASSES)).argmax(axis=-1)

    seg_img = np.zeros((int(HEIGHT/2), int(WIDTH/2),3))
    colors = class_colors

    for c in range(NCLASSES):
        seg_img[:,:,0] += ( (pr[:,: ] == c )*( colors[c][0] )).astype('uint8')
        seg_img[:,:,1] += ((pr[:,: ] == c )*( colors[c][1] )).astype('uint8')
        seg_img[:,:,2] += ((pr[:,: ] == c )*( colors[c][2] )).astype('uint8')

    seg_img = Image.fromarray(np.uint8(seg_img)).resize((orininal_w,orininal_h))

    image = Image.blend(old_img,seg_img,0.3)
    image.save(os.path.join(path, "%s_diejia.jpg" % jpg[ :-4]))
    seg_img.save(os.path.join(path, "%s_predict.jpg" % jpg[ :-4]))