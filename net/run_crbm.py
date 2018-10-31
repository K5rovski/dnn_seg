from yadlt.models.boltzmann import convrbm
import lmdb

# from testing_vis import draw_someoutput
from train_helper import iterate_unsup_indef, make_split_map
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import cv2
import os
import itertools
from skimage.io import imsave



img_h,img_w=45,45
image_size=(img_h,img_w,3)
hidden_maps=80
kernel_size=(4,4)
learning_rate=0.0008
batch_size=500
num_epochs=3
doSave=False
visible_unit_type='bin'
activation_type='softmax'


# ===@   Modifing code -signal modifying parameters often as a rule
save_pics_loc='crbm_80perc_normalprodset_normalsample_deep6_pics'# '########'
name='crbm_80perc_normalprodset_normalsample_deep'

model_loc=r'D:\data_projects\neuron_fibers_data\models'

# every 200x500 instances i.e. 100 000 instances
save_inepoch=200


if save_pics_loc is not None and not os.path.exists(save_pics_loc):
    os.makedirs(save_pics_loc)


r = convrbm.CONVRBM(image_size, hidden_maps,kernel_size,
                    visible_unit_type=visible_unit_type,
        name=name, loss_func='mse', learning_rate=learning_rate,
                    activation_type=activation_type,

                    model_separate_folder=model_loc,save_inepoch=save_inepoch,


        regcoef=5e-4, regtype='none', gibbs_sampling_steps=3,stddev=0.1,
            batch_size=batch_size, num_epochs=num_epochs,train_feedback=5)

# Fit the model
print('Start training...')



# ###########################################################################################
# train_lmdb='D:/NEURON_FIBER_DATA/lmdb_base_2k_45w_corrected_2mil_good/train_db'
# val_lmdb='D:/NEURON_FIBER_DATA/lmdb_base_2k_45w_corrected_2mil_good/val_db'

# train_lmdb='D:/NEURON_FIBER_DATA/lmdb_base_2k_45w_normal_mask_6milset_mixedset_80perc_20pics_prod/train_db'
# val_lmdb='D:/NEURON_FIBER_DATA/lmdb_base_2k_45w_mask_6milset_normalset_prod/val_db'
# #
#
#
#
#
# lmdbval_env=lmdb.open(val_lmdb)
# lmdbval_txn=lmdbval_env.begin()
#
# lmdbtrain_env=lmdb.open(train_lmdb)
# lmdbtrain_txn=lmdbtrain_env.begin()

# train_size=lmdbtrain_env.stat()['entries']#//5
# val_size=12100#lmdbval_env.stat()['entries']
#
# print('Training {} instances'.format(train_size))
#
# train_size=int(train_size/batch_size)
# val_size=int(val_size/batch_size)
# ###################################################################################




# --------------------------
#
# r.fit_generator(iterate_unsup_indef(lmdbtrain_txn,batch_size,img_w,img_h,do_continuous=True,
#                                     divide_some=1.0,split_map=True),train_size,
#
#         iterate_unsup_indef(lmdbval_txn,batch_size,img_w,img_h,do_continuous=True,
#                             divide_some=1.0,split_map=True),val_size)

# ---------------------------
#
rbm_model_loc=r'D:\data_projects\neuron_fibers_data\models\mixedset_6mil_4kernel_80perc_{}_ep0_valloss_2280.npy'
w_found = np.load(rbm_model_loc.format('Weight'))
vbias = np.load(rbm_model_loc.format('vbias'))
hbias = np.load(rbm_model_loc.format('hbias'))
r.initialize_weights((w_found,vbias,hbias))


# sumall=0.0
# sumallsq=0.0
# for patch in iterate_unsup_indef(lmdbtrain_txn,batch_size,img_w,img_h,do_continuous=False,divide_some=1.0,
#                                  smallify_base=train_size):
#     sumall+=np.average(patch.flatten())
#     sumallsq+=np.average((patch**2 ).flatten())
#
# all_avg=sumall/train_size
# all_avg_sq=sumallsq/train_size
# standard_dev=math.sqrt(all_avg_sq-all_avg**2)



# print('standard avg: ',all_avg,'standard_dev: ',standard_dev)

# lmdbtrain_env.close()

# real_vals=np.array(list(iterate_unsup_indef(lmdbval_txn,batch_size,
#                                    img_w,img_h,do_continuous=False,
#                                    smallify_base=val_size*batch_size,divide_some=1.0,split_map=True,
#                                             split_list=[0,128,255])))
#
# real_vals=real_vals.reshape( (-1,)+image_size )
#
# real_vals=real_vals[ np.random.choice(real_vals.shape[0],batch_size,replace=False)]
img=cv2.imread(r'D:\data_projects\neuron_fibers_data\images\test_spread_interim\sp13909-img04-interim.tif' ,0)
patch=img[834-22:834+23,676-22:676+23]

xcor,ycor=1352,922
xcor,ycor=1043,871
xcor,ycor=1180,1192
xcor,ycor=1458,1134
xcor,ycor=1465,934


patch=img[xcor-22:xcor+23,ycor-22:ycor+23]
real_vals=make_split_map(patch)





val_recs,hidden_probs,hidden_outs=r.reconstruct(real_vals,return_hidden=True)

# print('Some output',val_recs[:5,:2,:2])
# draw_someoutput()




val_recs_see=np.argmax(val_recs,axis=-1)

real_vals_see=np.argmax(real_vals,axis=-1)


if save_pics_loc is not None:
    for i in range(real_vals.shape[0]):

        cv2.imwrite( os.path.join(save_pics_loc, 'crbm_{}_real_pic.tif'.format(i))
                     ,(real_vals_see[i] *127).astype(np.uint8) )

        cv2.imwrite(os.path.join(save_pics_loc,'crbm_{}_restruction.tif'.format(i))
                    , (val_recs_see[i]*127).astype(np.uint8)  )

        for j in range(20):
             imsave(os.path.join(save_pics_loc, 'crbm_{}_hidden_{}.tif'.format(i,j))
                    ,((hidden_probs[0,:,:,j]/np.max(hidden_probs[:,:,:,j]))*255).astype(np.uint8),plugin='tifffile' )



# draw_someoutput(real_vals_see[:25],'crbm_maps_good6milset_80perc','visible_input')
# draw_someoutput(np.transpose(hidden_outs[3],(2,0,1)),'crbm_maps_good6milset_80perc','fourth_mat_hiddens2')

print(mean_squared_error(real_vals.reshape(real_vals.shape[0],-1),val_recs.reshape(val_recs.shape[0],-1)))


# score = model.evaluate_generator(
#     iterate_indef(lmdbval_txn, batch_size, img_w, img_h, do_continuous=True),
#         val_samples=val_size-val_size%batch_size, verbose=1)
#
# print('Test score:', score[0])
# print('Test accuracy:', score[1])


# lmdbval_env.close()