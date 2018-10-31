
import time
import lmdb
import sys
import os
import numpy as np

from keras.callbacks import ModelCheckpoint,EarlyStopping,LambdaCallback, TensorBoard
import keras.backend.tensorflow_backend as KTF



sys.path.append('../..')

from dnn_seg.net.utils.LogLossesCallback import LogLossesCallback
from dnn_seg.net.utils.models import get_autoencoded,get_autoencoded_singlein
from dnn_seg.net.utils.train_h import iterate_indef, quick_test, get_session, draw_onfirst_epoch,\
    save_relevant, use_best_model, use_num_model, \
    draw_autoenc, make_split_map, draw_autoenc_smaller





# GENERAL SETUP, SAVING THIS WORKSPACE ========================================


KTF.set_session(get_session(0.95))
np.random.seed(None)
quick_str=''.join(map(chr,np.random.randint(97,97+26,(4,))) )
weights_prepend='aenmodel_'+quick_str

files_save=[os.path.basename(__file__)]
config_text=save_relevant('saved_nnconfs', weights_prepend,files=files_save,just_return=True)




# ## AUTO ENCODER PARAMETER SETTING ================================================

DESCRIPTION_TEXT='Training a new AEN Instance'



train_lmdb=r'D:\data_projects\neuron_fibers_data\BASES\lmdb_base_2k_45w_autoenc_normal_6milset_topcont_interimn_neigh_circ_20pics_prod_reff\train_db'
val_lmdb=r'D:\data_projects\neuron_fibers_data\BASES\lmdb_base_2k_45w_autoenc_normal_6milset_topcont_interimn_neigh_circ_20pics_prod_reff\val_db'

model_locs=r'D:\data_projects\neuron_fibers_data\models'
log_loc=r'D:\data_projects\neuron_fibers_data\tensor_logs\new'
hist_loc=r'D:\data_projects\neuron_fibers_data\autoenc\history'


img_w,img_h=45,45
big_img_size=2048

optimizer='adadelta'


loss_func='categorical_crossentropy' 

batch_size=300
epoch_count=1
split_patches=True   #True for norml autoenc
do_single_map=False    #False for normal autoenc

model = get_autoencoded(img_w, img_h,)





# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>



save_model_call=ModelCheckpoint(os.path.join(model_locs,weights_prepend+'.{epoch:02d}-{val_loss:.4f}.hdf5'),
                                verbose=1,monitor='val_loss'
                                )
earlystop_call=EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=5, verbose=1, mode='auto')
tensor_call=TensorBoard(log_dir=log_loc, histogram_freq=3, write_graph=True, write_images=True)

log_losses=LogLossesCallback(123500//batch_size,1,model_id=quick_str,save_loc=hist_loc)



all_calls=[save_model_call,earlystop_call,tensor_call,log_losses]




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
train_size=500000#mdbtrain_env.stat()['entries']
val_size=83500



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

save_relevant('saved_nnconfs',quick_str,str_to_save=config_text,descriptive_text=DESCRIPTION_TEXT)


