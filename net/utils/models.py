from keras.models import Sequential,Model, load_model
from keras.layers import Convolution2D,MaxPooling2D,Input,Cropping2D,merge,Lambda, \
            Dense,Dropout,BatchNormalization,Activation, Flatten, Reshape, UpSampling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras import initializations
# from keras.utils.visualize_util import  plot
import numpy as np
import tensorflow as tf
import sys

sys.path.append('../../..')

from dnn_seg.net.utils.AutoEncoderLayer import AutoEncoderLayer
from dnn_seg.net.utils.CRBMPretrained import CRBMPretrained


def strez_init(shape, name=None,**kwargs):
    return initializations.normal(shape, scale= 0.00999999977648, name=name)

def get_pretrained(model_loc=
            r'C:\Users\kiko5\.yadlt\models\crbm_goodset_80perc_{}_ep0_valloss_3333.npy'):
    w_found=np.load(model_loc.format('Weight'))
    vbias=np.load(model_loc.format('vbias'))
    hbias = np.load(model_loc.format('hbias'))
    return w_found,vbias,hbias



def add_starting_bias(xtensor,added_bias=None):
    if added_bias is None:
        raise Exception('no added bias')

    return xtensor+np.array(added_bias)

def get_2k_image_pretrained(img_w,img_h,added_stuff=None,ch_add=3):
    model = Sequential()

    # w_crbm,vbias,hbias=get_pretrained()
    #
    # model.add(Lambda(add_starting_bias,
    #                  input_shape=(img_w, img_h, 3),arguments={'added_bias':list(vbias)}))
    #
    # model.add(Convolution2D(80,4,4,weights=(w_crbm,hbias),
    #                        input_shape=(img_w, img_h, 3), border_mode='valid'
    #                         , activation='sigmoid',W_learning_rate_multiplier=0.0,b_learning_rate_multiplier=0.0))
    #
    if added_stuff['use_crbm']:
        model.add(CRBMPretrained((42,42,80),added_stuff['path_pretrain'] ,input_shape=(img_w,img_h,3),
                                ))
        model.add(Activation('sigmoid'))
    elif  added_stuff['use_autoenc']:
        model.add(BatchNormalization(input_shape=(img_w + (8 - (img_w % 8)), \
                                                  img_h + (8 - (img_h % 8)), ch_add)))
        model.add(AutoEncoderLayer(added_stuff,))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
    else:
        model.add(BatchNormalization(input_shape=(img_w,img_h,1)))
        model.add(Convolution2D(80, 4, 4, init='glorot_normal', activation='linear', border_mode='same'))
        model.add(Activation('relu'))
        
        
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    # View this change  # model.add(Dropout(0.25))

    model.add(Convolution2D(70, 4, 4, init='glorot_normal', activation='linear', border_mode='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(60, 4, 4, init='glorot_normal', activation='linear', border_mode='same'))

    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same', ))

    model.add(Dropout(0.25))

    model.add(Flatten())

    # model.add(BatchNormalization())

    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3, init='glorot_normal', activation='softmax'))
    # model.add(Activation('softmax'))

    for i in range(len(model.layers)):
        print(model.layers[i].output_shape, model.layers[i].output_shape)

    return model

def get_2k_image_2layer_convnetmodel(img_w,img_h):
    model=Sequential()

    model.add(BatchNormalization(input_shape=(img_w,img_h,1)))
    model.add(Convolution2D(184,4,4,init='glorot_normal',input_shape=(img_w,img_h,1),border_mode='same'
                      ,activation='linear'      ))

    model.add(BatchNormalization ())
    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(92, 4,4, init='glorot_normal',activation='linear' ,border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.25))


    model.add(Convolution2D(60, 4, 4, init='glorot_normal',activation='linear',border_mode='same'))

    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same',))

    model.add(Dropout(0.4))

    model.add(Flatten())

    # model.add(BatchNormalization())

    # model.add(Dense(500))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))

    model.add(Dense(3,init='glorot_normal',activation='softmax'))
    # model.add(Activation('softmax'))

    for i in range(len(model.layers)):
        print(model.layers[i].output_shape, model.layers[i].output_shape)


    return model



def get_2k_image_good_convmodel(img_w,img_h):
    model=Sequential()

    model.add(BatchNormalization(input_shape=(img_w,img_h,3)))
    model.add(Convolution2D(184,4,4,init='glorot_normal',
                input_shape=(img_w,img_h,3),border_mode='same'
                      ,activation='linear'      ))


    model.add(Activation('relu'))


    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='same'))
    model.add(Dropout(0.25))

    model.add(Convolution2D(92, 4,4, init='glorot_normal',activation='linear' ,border_mode='same'))
    # model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same'))
    model.add(Dropout(0.25))


    model.add(Convolution2D(60, 4, 4, init='glorot_normal',activation='linear',border_mode='same'))

    # model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same',))

    model.add(Dropout(0.25))

    model.add(Flatten())

    # model.add(BatchNormalization())

    model.add(Dense(500))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(3,init='glorot_normal',activation='softmax'))
    # model.add(Activation('softmax'))

    for i in range(len(model.layers)):
        print(model.layers[i].output_shape, model.layers[i].output_shape)


    return model

def pad_one(x):
    # x_s=tf.shape(x)
    new_x=tf.pad(  x, [[0,0],[0,1],[0,1],[0,0]] )
    # new_x=tf.transpose(new_x,[0,3,1,2])
    return new_x

def get_autoencoded(imgw,imgh):
    input_img = Input(shape=(imgw + (8 - (imgw % 8)),
                             imgh + (8 - (imgh % 8)), 3))  # adapt this if using `channels_first` image data format

    # pad_img=Lambda(pad_one,  )(input_img)
    pad_img=BatchNormalization()(input_img)

    x = Convolution2D(80, 4, 4, activation='relu',border_mode='same')(pad_img)
    x=Dropout(0.5)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(x)

    x = Convolution2D(60, 4, 4, activation='relu', border_mode='same')(x)
    x = Dropout(0.33)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(x)
    #
    x = Convolution2D(40, 4, 4, activation='relu', border_mode='same')(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(x)

    encoded=x
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Convolution2D(40, 4, 4, activation='relu', border_mode='same')(encoded)
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(60, 4, 4, activation='relu', border_mode='same')(x)
    x = Dropout(0.33)(x)
    x = UpSampling2D((2, 2))(x)
    #
    x = Convolution2D(80, 4, 4, activation='relu', border_mode='same')(x)
    x = Dropout(0.5)(x)
    x = UpSampling2D((2, 2))(x)


    decoded = Convolution2D(3, 4, 4, activation='sigmoid', border_mode='same')(x)

    model = Model(input_img, decoded)
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model


def get_autoencoded_singlein(imgw,imgh,addIn=3):
    input_img = Input(shape=(imgw+addIn, imgh+addIn, 1))  # adapt this if using `channels_first` image data format

    normal_img=BatchNormalization()(input_img)
    # pad_img=Lambda(pad_one,  )(input_img)

    x = Convolution2D(80, 4, 4, activation='relu',border_mode='same')(normal_img)
    x=Dropout(0.5)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(x)

    x = Convolution2D(60, 4, 4, activation='relu', border_mode='same')(x)
    x = Dropout(0.33)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(x)
    #
    x = Convolution2D(40, 4, 4, activation='relu', border_mode='same')(x)
    x = Dropout(0.25)(x)
    x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(x)

    encoded=x
    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    x = Convolution2D(40, 4, 4, activation='relu', border_mode='same')(encoded)
    x = Dropout(0.25)(x)
    x = UpSampling2D((2, 2))(x)

    x = Convolution2D(60, 4, 4, activation='relu', border_mode='same')(x)
    x = Dropout(0.33)(x)
    x = UpSampling2D((2, 2))(x)
    #
    x = Convolution2D(80, 4, 4, activation='relu', border_mode='same')(x)
    x = Dropout(0.5)(x)
    x = UpSampling2D((2, 2))(x)


    decoded = Convolution2D(3, 4, 4, activation='sigmoid', border_mode='same')(x)

    model = Model(input_img, decoded)
    model.compile(optimizer='adadelta', loss='binary_crossentropy')
    return model

def get_2k_twopath_simple(img_w,img_h):
    input_img=Input(shape=(img_w,img_h,1))

    normal=BatchNormalization(input_shape=(img_w,img_h,1))(input_img)

    conv1=Convolution2D(184,4,4,init='glorot_normal',input_shape=(img_w,img_h,1),border_mode='valid'
                      ,activation='relu'      )(normal)
    max1=MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='same')(conv1)
    drop1=Dropout(0.25)(max1)

    normal2=BatchNormalization()(drop1)

    conv2=Convolution2D(60, 4,4, init='glorot_normal',activation='relu' ,border_mode='valid')(normal2)
    max2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(conv2)
    drop2=Dropout(0.25)(max2)


    conv3=Convolution2D(40, 4, 4, init='glorot_normal',activation='relu',border_mode='same')(drop2)
    max3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(conv3)
    drop3 = Dropout(0.25)(max3)



    smaller=Cropping2D(cropping=((12,12), (12,12)))(normal)

    smaller_still = Cropping2D(cropping=((18, 18), (18, 18)))(normal)

    conv1_small=Convolution2D(30, 4,4, init='glorot_normal',activation='relu' ,border_mode='valid')(smaller)
    max1_small = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(conv1_small)
    drop1_small = Dropout(0.25)(max1_small)


    conv2_small = Convolution2D(20, 4, 4, init='glorot_normal',
                    activation='relu', border_mode='same')(drop1_small)
    max2_small = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(conv2_small)
    drop2_small = Dropout(0.25)(max2_small)


    merge_all=merge([drop3,drop2_small],mode='concat',concat_axis=-1)

    flat=Flatten()(merge_all)

    # model.add(BatchNormalization())

    dense1=Dense(500,activation='relu')(flat)
    drop_final=Dropout(0.5)(dense1)

    choice=Dense(3,init='glorot_normal',activation='softmax')(drop_final)
    # model.add(Activation('softmax'))

    model=Model(input=input_img,output=choice)

    for i in range(len(model.layers)):
        print(model.layers[i].output_shape, model.layers[i].output_shape)

    plot(model, to_file='model_twopath.png')

    return model

def get_2k_twopath_twoconv(img_w,img_h):
    input_img=Input(shape=(img_w,img_h,1))

    normal=BatchNormalization(input_shape=(img_w,img_h,1))(input_img)

    conv1=Convolution2D(184,4,4,init='glorot_normal',input_shape=(img_w,img_h,1),border_mode='valid'
                      ,activation='relu'      )(normal)
    max1=MaxPooling2D(pool_size=(2, 2), strides=(2,2), border_mode='same')(conv1)
    drop1=Dropout(0.25)(max1)

    conv2=Convolution2D(60, 4,4, init='glorot_normal',activation='relu' ,border_mode='valid')(drop1)
    max2=MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(conv2)
    drop2=Dropout(0.25)(max2)



    conv1_bigk=Convolution2D(184,8,8,init='glorot_normal',input_shape=(img_w,img_h,1),border_mode='valid'
                      ,activation='relu'      )(normal)
    max1_bigk = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(conv1_bigk)
    drop1_bigk = Dropout(0.25)(max1_bigk)



    merge_all=merge([drop2,drop1_bigk],mode='concat',concat_axis=-1)

    conv3=Convolution2D(40, 4, 4, init='glorot_normal',activation='relu',border_mode='same')(merge_all)
    max3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), border_mode='same')(conv3)
    drop3 = Dropout(0.25)(max3)

    flat=Flatten()(drop3)

    # model.add(BatchNormalization())

    dense1=Dense(500,activation='relu')(flat)
    drop_final=Dropout(0.5)(dense1)

    choice=Dense(3,init='glorot_normal',activation='softmax')(drop_final)
    # model.add(Activation('softmax'))

    model=Model(input=input_img,output=choice)

    for i in range(len(model.layers)):
        print(model.layers[i].output_shape, model.layers[i].output_shape)

    plot(model, to_file='model_twopath.png')

    return model
