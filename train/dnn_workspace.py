import os
import time
from functools import partial

import lmdb
import sys
import tensorflow as tf
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard,EarlyStopping,LambdaCallback
from keras.models import load_model

from keras.optimizers import SGD
import keras.backend.tensorflow_backend as KTF

import AutoEncoderLayer
from CRBMPretrained import CRBMPretrained
from LogLossesCallback import LogLossesCallback
from train_helper import iterate_indef
from net_models import get_2k_image_2layer_convnetmodel
import numpy as np
from numpy import array as arr

from train_helper import quick_test,get_session,draw_onfirst_epoch,save_relevant,use_best_model,use_num_model

from net_models import get_2k_image_good_convmodel,get_2k_twopath_simple,get_2k_twopath_twoconv,get_2k_image_pretrained

np.random.seed(None)
KTF.set_session(get_session(0.8))


model_locs=r'D:\data_projects\neuron_fibers_data\models'
log_loc=r'D:\data_projects\neuron_fibers_data\tensor_logs'


val_lmdb=r'D:\data_projects\neuron_fibers_data\BASES\special_choice_8mil_45_Wsize_20imgs_interimONset\val_db'
train_lmdb=r'D:\data_projects\neuron_fibers_data\BASES\special_choice_8mil_45_Wsize_20imgs_interimONset\train_db'

img_w,img_h=45,45
big_img_size=2400

# optimizer = SGD(lr=0.0001, decay=0.0005, momentum=0.9, nesterov=True)
optimizer='adadelta'

loss_func='categorical_crossentropy' #'categorical_crossentropy'

batch_size=500
epoch_count=3


quick_str=''.join(map(chr,np.random.randint(97,97+26,(4,))) )

# False for new Training
continueTrain=True


justTestPreviousFullSet=True


split_patches=True  # true for autoenc

single_map_autoenc=False #true for single autoenc
numchannel_autoenc=3 # true for..1


epoch_toDraw=None # None for best epoch
start_patch48=True  # true for autoenc



added_stuff={
            # 'path_pretrain':
            #      r'C:\Users\kiko5\.yadlt\models\crbm_80perc_normalprodset_normalsample_{}_ep0_valloss_3333.npy',
    'use_crbm':False,
    'use_autoenc':True,
             'autoenc_loc':r'D:\data_projects\neuron_fibers_data\models\weights_autoenc_itjs.00-0.2338.hdf5',
             'autoenc_layer':'convolution2d_1'}
test_parDikt={
    'patch_size':45,
    'img_path':r'D:\data_projects\neuron_fibers_data\images\intsall\\',
    'groundtruth':r'D:\data_projects\neuron_fibers_data\images\cors\c\\',
    'interim':r'D:\data_projects\neuron_fibers_data\images\intsall\\'}


train_autoenc_loc=r'D:\data_projects\neuron_fibers_data\autoenc'
reduce_valSize_Fac=0.2 # 1

do_quick_after_train=False
do_test_after_one=False
test_after_train_fullset=False


do_whole_pic=True
draw_img_ext='*.tif'
val_freq=122300


weights_prepend='weights_quickbig_'+quick_str


print('Model is: ',weights_prepend)



if continueTrain:
    print('Continued Training')
    save_model_loc = r'D:\data_projects\neuron_fibers_data\models\weights_quickbig_fosd.00-0.8676.hdf5'

    # model=load_model(save_model_loc,custom_objects={'CRBMPretrained':CRBMPretrained})
    model=load_model(save_model_loc,custom_objects={'AutoEncoderLayer':AutoEncoderLayer.AutoEncoderLayer})

    quick_oldstr=save_model_loc[save_model_loc.rindex('weights_quickbig')+15:save_model_loc.rindex('hdf5')]
    # model_epoch=int(save_model_loc[save_model_loc.rindex(quick_oldstr)+4 +1:save_model_loc.rindex(quick_oldstr)+4 +3])

    quick_oldstr+='mlset_interimON_8mil_ints_special_pretrain_normalized'

    if justTestPreviousFullSet:
        for older,newer,xx in zip(range(0,62,3),range(3,62,3),range(100)):

            test_parDikt.update({'from': older, 'to': newer})

            quick_test(model, quick_oldstr, big_img_size, do_all=do_whole_pic,
                       draw_img_ext=draw_img_ext, test_batch_size=300,
                       split_map=split_patches,
                       conf_dikt=test_parDikt,two_patch_instance=start_patch48)
        sys.exit('I\'m done continued testing')
else:
    print('New Training')
    model = get_2k_image_pretrained(img_w, img_h,added_stuff,ch_add=numchannel_autoenc)

    model_epoch=0
    model.compile(loss=loss_func,
                  optimizer=optimizer,
                  metrics=['accuracy'])

config_text=save_relevant('saved_quickmodels', quick_str,just_return=True)



# reduce_lr_call=ReduceLROnPlateau(monitor='val_acc',factor=0.2,
#                                  patience=3,cooldown=2,verbose=1)
save_model_call=ModelCheckpoint(os.path.join(model_locs,weights_prepend+'.{epoch:02d}-{val_acc:.4f}.hdf5'),
                                verbose=1,monitor='val_acc'
                                )
earlystop_call=EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=5, verbose=1, mode='auto')
# tensor_call=TensorBoard(log_dir=log_loc, histogram_freq=3, write_graph=True, write_images=True)





dodraw_afterone=LambdaCallback(
    on_epoch_end=partial(draw_onfirst_epoch,
                         model=model,
                         big_img_size=big_img_size,
                         do_test=do_test_after_one,
                         quick_str=quick_str,
                         str_to_save=config_text,
                         split_map=split_patches))



log_losses=LogLossesCallback(val_freq//batch_size,(val_freq*3)//batch_size,model_id=quick_str,
                             save_loc=train_autoenc_loc,save_model=r'D:\data_projects\neuron_fibers_data\autoenc')

all_calls=[save_model_call,earlystop_call,dodraw_afterone]







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
val_size=int(lmdbval_env.stat()['entries']*reduce_valSize_Fac)


oneI=iterate_indef(lmdbval_txn, batch_size, img_w *2 if start_patch48 else img_w, img_h, two_patch_instance=start_patch48,
              do_continuous=True,
                   do_single_patch=single_map_autoenc,
                   split_map=split_patches,return_dnn_annotation=True)

raw_patches=[next(oneI) for ind in range(4)]
val_data_x,val_data_y=arr([i[0] for i in raw_patches]),arr([i[1] for i in raw_patches])
oneI=None
log_losses.val_data=val_data_x.reshape((4*batch_size,)+val_data_x.shape[2:]),val_data_y.reshape((4*batch_size,)+val_data_y.shape[2:])

model.fit_generator(
    iterate_indef(lmdbtrain_txn,batch_size,img_w *2 if start_patch48 else img_w,img_h,
                  do_continuous=True,
        do_single_patch=single_map_autoenc,
                  split_map=split_patches,two_patch_instance=start_patch48,return_dnn_annotation=True),
            samples_per_epoch=train_size-train_size%batch_size,
                    nb_epoch=epoch_count,

          verbose=1,
        callbacks=all_calls,
        validation_data= \
        iterate_indef(lmdbval_txn, batch_size, img_w *2 if start_patch48 else img_w, img_h,
                    do_single_patch=single_map_autoenc,
                      do_continuous=True,split_map=split_patches,two_patch_instance=start_patch48,return_dnn_annotation=True),
        nb_val_samples=val_size-val_size%batch_size )


lmdbtrain_env.close()

# score = model.evaluate_generator(
#     iterate_indef(lmdbval_txn, batch_size, img_w, img_h, do_continuous=True),
#         val_samples=val_size-val_size%batch_size, verbose=1)
#
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


lmdbval_env.close()


if do_quick_after_train:
    best_model_loc=use_num_model(quick_str,num_model=epoch_toDraw)
    best_model=load_model(best_model_loc,custom_objects={'CRBMPretrained':CRBMPretrained})

    quick_test(best_model,quick_str,big_img_size,do_all=do_whole_pic
               ,split_map=split_patches,draw_img_ext=draw_img_ext)
    # save_relevant('saved_quickmodels',quick_str)

if test_after_train_fullset:

    quick_str += 'fullset'
    for older, newer in zip(range(0, 31, 8), range(8, 33, 8)):
        test_parDikt.update({'from':older,'to':newer})

        quick_test(model, quick_str, big_img_size, do_all=do_whole_pic, draw_img_ext=draw_img_ext, test_batch_size=500,
                   split_map=split_patches,
                   conf_dikt=test_parDikt)

print('time duration is: ',time.time()-start_t)


