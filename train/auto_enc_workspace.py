from functools import partial

import time

import cv2
import lmdb
import sys
from keras.backend import get_session
from keras.callbacks import ModelCheckpoint,EarlyStopping,LambdaCallback, TensorBoard
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from keras.models import Model, load_model
from keras import backend as K
import keras.backend.tensorflow_backend as KTF

from LogLossesCallback import LogLossesCallback
from net_models import get_autoencoded,get_autoencoded_singlein
from train_helper import iterate_indef, quick_test, get_session, draw_onfirst_epoch, save_relevant, use_best_model, \
    use_num_model, \
    draw_autoenc, make_split_map, draw_autoenc_smaller

import os
import pickle
import tensorflow as tf
from os.path import join
import numpy as np



KTF.set_session(get_session(0.8))

train_lmdb=r'D:\data_projects\neuron_fibers_data\BASES\special_choice_8mil_45_Wsize_20imgs_interimONset\train_db'
val_lmdb=r'D:\data_projects\neuron_fibers_data\BASES\special_choice_8mil_45_Wsize_20imgs_interimONset\val_db'

model_locs=r'D:\data_projects\neuron_fibers_data\models'
log_loc=r'D:\data_projects\neuron_fibers_data\tensor_logs'
img_w,img_h=45,45
big_img_size=2400

# optimizer = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
optimizer='adadelta'


loss_func='categorical_crossentropy' #'categorical_crossentropy'

batch_size=500
epoch_count=1
split_patches=True   #True for norml autoenc
do_single_map=False    #False for normal autoenc

np.random.seed(None)
quick_str=''.join(map(chr,np.random.randint(97,97+26,(4,))) )

weights_prepend='weights_autoenc_'+quick_str


#!!!!!!!!!!!! Just drawing inside results
doJustDrawEnc=True

model_loc=r'D:\data_projects\neuron_fibers_data\models\weights_autoenc_dvov.00-0.2944.hdf5'
draw_encoded_loc=r'D:\code_projects\dnn_seg\autoenc\encoded_maps_vis'
hist_loc=r'D:\data_projects\neuron_fibers_data\autoenc\history'

save_model_call=ModelCheckpoint(os.path.join(model_locs,weights_prepend+'.{epoch:02d}-{val_loss:.4f}.hdf5'),
                                verbose=1,monitor='val_loss'
                                )
earlystop_call=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
tensor_call=TensorBoard(log_dir=log_loc, histogram_freq=3, write_graph=True, write_images=True)

log_losses=LogLossesCallback(123500//batch_size,1,model_id=quick_str,save_loc=hist_loc)
# LogLossesCallback()



all_calls=[save_model_call,earlystop_call,tensor_call,log_losses]


if doJustDrawEnc:
    model=load_model(model_loc,custom_objects={'tf':tf})
    mod_name=model_loc[model_loc.index('weights_autoenc_'):]
    draw_loc=join(draw_encoded_loc,mod_name)
    if not os.path.exists(draw_loc): os.makedirs(draw_loc)

    patchs=[]

    img1 = cv2.imread(r'D:\data_projects\neuron_fibers_data\images\test_spread_interim\sp13909-img04-interim.tif', 0)
    img2=cv2.imread(r'D:\data_projects\neuron_fibers_data\images\all_test_big_corrected\sp13909-img04-corrected.tif',0)
    for img in [img1,img2]:
        patchs.append( img[834 - 22:834 + 23, 676 - 22:676 + 23])

        xcor, ycor = 1352, 922
        patchs.append(img[xcor - 22:xcor + 23, ycor - 22:ycor + 23])
        xcor, ycor = 1043, 871
        patchs.append(img[xcor - 22:xcor + 23, ycor - 22:ycor + 23])
        xcor, ycor = 1180, 1192
        patchs.append(img[xcor - 22:xcor + 23, ycor - 22:ycor + 23])
        xcor, ycor = 1458, 1134
        patchs.append(img[xcor - 22:xcor + 23, ycor - 22:ycor + 23])
        xcor, ycor = 1465, 934
        patchs.append(img[xcor - 22:xcor + 23, ycor - 22:ycor + 23])


    patchs=np.array(patchs)


    real_vals = make_split_map(patchs[:6],dopad=True)
    real_vals2=make_split_map(patchs[6:],dopad=True,split_map=[1,2,3])
    patches = np.zeros((2, 6, 48, 48, 3), dtype=np.uint8)
    patches[0,:]=real_vals
    patches[1,:]=real_vals2

    draw_autoenc_smaller(model,draw_loc,patches)

#    draw_autoenc(model, img_w, img_h, draw_loc, val_lmdb)
    sys.exit('Im am done with everything!!')

else:
    model = get_autoencoded(img_w, img_h,)

# ====================================================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# ====================================================



start_t=time.time()


lmdbval_env=lmdb.open(val_lmdb)
lmdbval_txn=lmdbval_env.begin()

lmdbtrain_env=lmdb.open(train_lmdb)
lmdbtrain_txn=lmdbtrain_env.begin()
train_size=lmdbtrain_env.stat()['entries']
val_size=83500#lmdbval_env.stat()['entries']

oneI=iterate_indef(lmdbval_txn, batch_size, img_w * 2, img_h, two_patch_instance=True,
              do_continuous=True, split_map=split_patches,do_single_patch=do_single_map)
raw_patches=[next(oneI) for ind in range(5)]
val_data_x,val_data_y=np.concatenate([i[0] for i in raw_patches]), \
                      np.concatenate([i[1] for i in raw_patches])
oneI=None
log_losses.val_data=val_data_x,val_data_y

history_mod=model.fit_generator(
    iterate_indef(lmdbtrain_txn,batch_size,img_w*2,img_h,
                  do_continuous=True,split_map=split_patches,
                  do_single_patch=do_single_map,two_patch_instance=True,),
            samples_per_epoch=train_size-train_size%batch_size,
                    nb_epoch=epoch_count,

          verbose=1,
        callbacks=all_calls,
        validation_data= \
        iterate_indef(lmdbval_txn, batch_size, img_w*2, img_h,two_patch_instance=True,
                      do_single_patch=do_single_map,do_continuous=True,split_map=split_patches),
        nb_val_samples=val_size-val_size%batch_size )


lmdbtrain_env.close()

# score = model.evaluate_generator(
#     iterate_indef(lmdbval_txn, batch_size, img_w, img_h, do_continuous=True),
#         val_samples=val_size-val_size%batch_size, )
# #
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


lmdbval_env.close()


