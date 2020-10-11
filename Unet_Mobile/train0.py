from nets.unet import mobilenet_unet
from keras.optimizers import Adam
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from PIL import Image
import keras
from keras import backend as K
import numpy as np
from osgeo import gdal,gdal_array

NCLASSES = 2
HEIGHT = 416
WIDTH = 416


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
            # 从文件中读取图像
            
            img = gdal.Open(r"../mydatargb/tif" + '/' + name)  ##00改
            im_width = img.RasterXSize  # 栅格矩阵的列数   ##00改
            im_height = img.RasterYSize  # 栅格矩阵的行数   ##00改
            im_bands = img.RasterCount  # 波段数        ##00改
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
            img = img.ReadAsArray(0,0,im_width,im_height,None,int(WIDTH/2),int(HEIGHT/2)) # 获取数据
            img = np.transpose(img,(1,2,0))
            #
            seg_labels = np.zeros((int(HEIGHT/2),int(WIDTH/2),NCLASSES))
            for c in range(NCLASSES):
                seg_labels[: , : , c ] = (img[:,:,0] == c ).astype(int)
            seg_labels = np.reshape(seg_labels, (-1,NCLASSES))
            Y_train.append(seg_labels)


            # 读完一个周期后重新开始
            i = (i+1) % n
        yield (np.array(X_train),np.array(Y_train))

def loss(y_true, y_pred):
    crossloss = K.binary_crossentropy(y_true,y_pred)
    loss = 4 * K.sum(crossloss)/HEIGHT/WIDTH
    return loss


def IoU(y_true, y_pred, eps=1e-6):
    if np.max(y_true) == 0.0:
        return IoU(1-y_true, 1-y_pred) ## empty image; calc IoU of zeros
    intersection = K.sum(y_true * y_pred, axis=[1,2,3])
    union = K.sum(y_true, axis=[1,2,3]) + K.sum(y_pred, axis=[1,2,3]) - intersection
    return -K.mean( (intersection + eps) / (union + eps), axis=0)
    "https://blog.csdn.net/wangdongwei0/article/details/82563689"

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
    model = mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
    # model.summary()
    BASE_WEIGHT_PATH = ('https://github.com/fchollet/deep-learning-models/'
										'releases/download/v0.6/')
    model_name = 'mobilenet_%s_%d_tf_no_top.h5' % ( '1_0' , 224 )
   
    weight_path = BASE_WEIGHT_PATH + model_name
    weights_path = keras.utils.get_file(model_name, weight_path )
    print(weight_path)
    model.load_weights(weights_path,by_name=True,skip_mismatch=True)

    # model.summary()
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

    # 保存的方式，5世代保存一次
    checkpoint_period = ModelCheckpoint(
                                    log_dir + 'ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5',
                                    monitor='val_loss', 
                                    save_weights_only=True, 
                                    save_best_only=True, 
                                    period=5
                                )
    # 学习率下降的方式，val_loss三次不下降就下降学习率继续训练
    reduce_lr = ReduceLROnPlateau(
                            monitor='val_loss', 
                            factor=0.5, 
                            patience=3, 
                            verbose=1
                        )
    # 是否需要早停，当val_loss一直不下降的时候意味着模型基本训练完毕，可以停止
    early_stopping = EarlyStopping(
                            monitor='val_loss', 
                            min_delta=0, 
                            patience=10, 
                            verbose=1
                        )

    # 交叉熵
    model.compile(loss = loss,
            optimizer = Adam(lr=1e-3),
            metrics = ['accuracy'])
    batch_size = 2
    print('Train on {} samples, val on {} samples, with batch size {}.'.format(num_train, num_val, batch_size))
    
    # 开始训练
    model.fit_generator(generate_arrays_from_file(lines[:num_train], batch_size),
            steps_per_epoch=max(1, num_train//batch_size),
            validation_data=generate_arrays_from_file(lines[num_train:], batch_size),
            validation_steps=max(1, num_val//batch_size),
            epochs=100,
            initial_epoch=0,
            callbacks=[checkpoint_period, reduce_lr])

    model.save_weights(log_dir+'last1.h5')