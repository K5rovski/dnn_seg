import os
import time
from functools import partial
import lmdb
import sys

from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint, TensorBoard,EarlyStopping,LambdaCallback
from keras.models import load_model

from keras.optimizers import SGD
import keras.backend.tensorflow_backend as KTF



import numpy as np





sys.path.append('../..')


from dnn_seg.net.utils.train_h import quick_test,get_session,draw_onfirst_epoch,save_relevant,use_best_model,use_num_model,iterate_indef
from dnn_seg.net.utils.models import get_2k_image_good_convmodel,get_2k_twopath_simple,get_2k_twopath_twoconv,get_2k_image_pretrained



# GENERAL SETUP, SAVING THIS WORKSPACE ========================================
np.random.seed(None)
KTF.set_session(get_session(0.8))
quick_str=''.join(map(chr,np.random.randint(97,97+26,(4,))) )

files_save=[os.path.basename(__file__)]

config_text=save_relevant('saved_nnconfs', quick_str,files=files_save,just_return=True)










# ## DNN PARAMETER SETTING ================================================

DESCRIPTION_TEXT='Training a new DNN Instance'




model_locs=r'D:\data_projects\neuron_fibers_data\models'
log_loc=r'D:\data_projects\neuron_fibers_data\tensor_logs'


val_lmdb=r'D:\data_projects\neuron_fibers_data\BASES\6mil_45pix_20imgs_interimTC15Neighset_10circ_prod_reff\val_db'
train_lmdb=r'D:\data_projects\neuron_fibers_data\BASES\6mil_45pix_20imgs_interimTC15Neighset_10circ_prod_reff\train_db'

img_w,img_h=45,45
big_img_size=2048

optimizer='adadelta'

loss_func='categorical_crossentropy' 

batch_size=500
epoch_count=3





split_patches=True  # true for autoenc

single_map_autoenc=False #true for single autoenc
numchannel_autoenc=3 # true for..1


epoch_toDraw=None # None for best epoch
start_double_patch=False  #  whether input data has two patches in x


added_stuff={
          
    'use_crbm':False,
    'use_autoenc':True,
             'autoenc_loc':r'D:\data_projects\neuron_fibers_data\models\weights_autoenc_itjs.00-0.2338.hdf5',
             'autoenc_layer':'convolution2d_1'}


reduce_valSize_Fac=0.2

do_test_after_one=False


do_whole_pic=True
draw_img_ext='*.tif'
val_freq=122300


# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>






weights_prepend='dnnmodel_'+quick_str
print('New Model is: ',weights_prepend)


model = get_2k_image_pretrained(img_w, img_h,added_stuff,ch_add=numchannel_autoenc)

model_epoch=0
model.compile(loss=loss_func,
			  optimizer=optimizer,
			  metrics=['accuracy'])


			  
			  
			  
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


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




all_calls=[save_model_call,earlystop_call,dodraw_afterone]








# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



start_t=time.time()


lmdbval_env=lmdb.open(val_lmdb)
lmdbval_txn=lmdbval_env.begin()

lmdbtrain_env=lmdb.open(train_lmdb)
lmdbtrain_txn=lmdbtrain_env.begin()
train_size=lmdbtrain_env.stat()['entries']
val_size=int(lmdbval_env.stat()['entries']*reduce_valSize_Fac)





model.fit_generator(
    iterate_indef(lmdbtrain_txn,batch_size,img_w ,img_h,
                  do_continuous=True,
        do_single_patch=single_map_autoenc,
                  split_map=split_patches,two_patch_instance=start_double_patch,do_padSq=3, dnn_class_annot=True),

        validation_data= \
        iterate_indef(lmdbval_txn, batch_size, img_w , img_h,
                    do_single_patch=single_map_autoenc,
                      do_continuous=True,split_map=split_patches,two_patch_instance=start_double_patch, \
                    do_padSq=3, dnn_class_annot=True),

            samples_per_epoch=train_size-train_size%batch_size,
                    nb_epoch=epoch_count,

          verbose=1,
        callbacks=all_calls,
        nb_val_samples=val_size-val_size%batch_size )


lmdbtrain_env.close()

# score = model.evaluate_generator(
#     iterate_indef(lmdbval_txn, batch_size, img_w, img_h, do_continuous=True),
#         val_samples=val_size-val_size%batch_size, verbose=1)
#
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


lmdbval_env.close()


print('time duration is: ',time.time()-start_t)

save_relevant('saved_nnconfs',quick_str,str_to_save=config_text,descriptive_text=DESCRIPTION_TEXT)

