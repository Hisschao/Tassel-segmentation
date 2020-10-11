from nets.pspnet import mobilenet_pspnet
import numpy as np

model = mobilenet_pspnet(2,576,576)
model.summary()