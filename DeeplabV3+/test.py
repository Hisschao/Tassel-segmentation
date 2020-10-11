from nets.deeplab import Deeplabv3
model = Deeplabv3(input_shape=(512*2, 512*2, 3),classes=2,OS=16)
model.summary()

