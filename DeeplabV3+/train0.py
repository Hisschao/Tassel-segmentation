from nets.deeplab import Deeplabv3
from keras.utils.data_utils import get_file
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import keras
from keras import backend as K
import numpy as np
from osgeo import gdal,gdal_array
import tensorflow as tf

ALPHA = 1.0
WEIGHTS_PATH_X = "https://github.com/bonlime/keras-deeplab-v3-plus/releases/download/1.1/deeplabv3_xception_tf_dim_ordering_tf_kernels.h5"
                    
NCLASSES = 2
HEIGHT = 512
WIDTH = 512


def generate_arrays_from_file(lines,batch_size):
    # 获取总长度
    n = len(lines)
    i = 0
    while 1:
        X_train = []
        Y_train = []
        # 获取一个batch_size大小的数据
        for _ in range(batch_size):
            if i==0:
                np.random.shuffle(lines)
            name = lines[i].split(';')[0]
            # 从文件中读取原图像
            img = gdal.Open(r"../mydatargb/tif" + '/' + name)  ##00改
            im_width = img.RasterXSize  # 栅格矩阵的列数   ##00改
            im_height = img.RasterYSize  # 栅格矩阵的行数   ##00改
            # im_bands = img.RasterCount  # 波段数        ##00改
            img = img.ReadAsArray(0, 0, im_width, im_height, None, WIDTH, HEIGHT)  # 00获取数据
            img = np.transpose(img,(1,2,0))            ##00gai
            img = img/255
            X_train.append(img)

            name = (lines[i].split(';')[1]).replace("\n", "")
            # 从文件中读取mask图像
            #gdal
            img = gdal.Open(r"../mydatargb/jpg" + '/' + name)
            im_width = img.RasterXSize  # 栅格矩阵的列数
            im_height = img.RasterYSize  # 栅格矩阵的行数
            img = img.ReadAsArray(0,0,im_width,im_height,None,int(WIDTH),int(HEIGHT)) # 获取数据
            img = np.transpose(img,(1,2,0))
            #
            seg_labels = np.zeros((int(HEIGHT),int(WIDTH),NCLASSES))
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y_train.append(seg_labels)

            # 读完一个周期后重新开始
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))

def loss(y_true, y_pred):
    crossloss = K.binary_crossentropy(y_true,y_pred)
    loss = K.sum(crossloss)/HEIGHT/WIDTH
    return loss

##CE
class WeightedBinaryCrossEntropy(object):
    def __init__(self, pos_ratio):
        neg_ratio = 1. - pos_ratio
        self.pos_ratio = tf.constant(pos_ratio, tf.float32)
        self.weights = tf.constant(neg_ratio / pos_ratio, tf.float32)
        self.__name__ = "weighted_binary_crossentropy({0})".format(pos_ratio)

    def __call__(self, y_true, y_pred):
        return self.weighted_binary_crossentropy(y_true, y_pred)

    def weighted_binary_crossentropy(self, y_true, y_pred):
        # Transform to logits         
        epsilon = tf.convert_to_tensor(K.common._EPSILON, y_pred.dtype.base_dtype)
        y_pred = tf.clip_by_value(y_pred, epsilon, 1 - epsilon)
        y_pred = tf.log(y_pred / (1 - y_pred))

        cost = tf.nn.weighted_cross_entropy_with_logits(y_true, y_pred, self.weights)
        return K.mean(cost * self.pos_ratio, axis=-1)
#meaniou
def meaniou(y_true, y_pred, smooth=0.0001):
    y_pred_ = K.round(y_pred)
    y_true_ = K.round(y_true)
    overlap = K.sum(K.abs(y_pred_ * y_true_))
    overlap0 = K.sum(K.abs((y_pred_ - 1) * (y_true_ - 1)))
    union = K.sum(K.clip(K.abs(y_pred_ + y_true_), 0, 1))
    union0 = K.sum(K.clip(K.abs(y_pred_ + y_true_ - 2), 0, 1))
    IOU = (overlap+smooth)/(union+smooth) + (overlap0+smooth)/(union0+smooth)
    return IOU/2

####  pa
def _true_positives(y_true, y_pred):
    # determine number of matching values and return the sum
    return K.sum(y_true * K.round(y_pred))
def _false_negatives(y_true, y_pred):
    return K.sum(K.clip((y_true - K.round(y_pred)), 0, 1))
def precision(y_true, y_pred):
    # Precision = TP / (TP + FP)
    tp = _true_positives(y_true, y_pred)
    return tp / (tp + _false_negatives(y_true, y_pred))

if __name__ == "__main__":
    log_dir = "logs/"
    # 获取model
    model = Deeplabv3(classes=2,input_shape=(HEIGHT,WIDTH,3))
    # model.summary()

    weights_path = get_file('deeplabv3_xception_tf_dim_ordering_tf_kernels.h5',
                                    WEIGHTS_PATH_X,
                                    cache_subdir='models')
    model.load_weights(weights_path,by_name=True)

    # 打开数据集的txt
    with open(r"../mydatargb/train_data.txt","r") as f:
        lines = f.readlines()

    # 打乱行，这个txt主要用于帮助读取数据来训练
    # 打乱的数据更有利于训练
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)

    # 90%用于训练，10%用于估计。
    num_val = int(len(lines)*0.1)
    num_train = len(lines) - num_val

    # 保存的方式，3世代保存一次  # 26  71
    checkpoint_period = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', 
                                    save_weights_only=True, 
                                    save_best_only=True, 
                                    period=1
                                )
    # 学习率下降的方式，acc三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', 
                            factor=0.5, 
                            patience=20, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )

    # 交叉熵 WeightedBinaryCrossEntropy(pos_ratio = 0.044850)
    model.compile(loss = WeightedBinaryCrossEntropy(pos_ratio = 0.044850),
            optimizer = Adam(lr=1e-4),
            metrics = [precision])  ##precision'accuracy'  meaniou
            
    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=200,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr])

    model.save_weights(log_dir+'last1.h5')
