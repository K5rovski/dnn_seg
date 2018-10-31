import numpy as np
np.random.seed(8)

import os
import sys

sys.path.append('../../..')

from dnn_seg.data_prep.utils.data_helper import get_patches



# Sample script to create some patches =======================
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>


lookup_test_path =r'D:\data_projects\neuron_fibers_data\TCINT_neigh15_circ10'
groundtruth_path = r'D:\data_projects\neuron_fibers_data\cors_mlset01'
patch_size=45

which_pix=\
'''654,1274
653,1272
659,1264
660,1266
671,1355
671,1349
681,1345
681,1357
681,1341
666,183
670,188
673,189
696,136
705,107
'''.splitlines()
# subsample_mask[which_pix[:,1],which_pix[:,0]]=True


subsample_mask=np.zeros((2048,2048),dtype=np.uint8)
sizable_input=(np.random.rand(2048,2048) <0.03)
print('Getting {} patches from image'.format(np.sum(sizable_input)))
subsample_mask[sizable_input]=True


subsample_mask=subsample_mask.flatten()

activs1=np.random.choice(80,80,replace=False)




save_out={}
save_out['activation_1']=np.zeros((80,48,48))
save_out['activation_2']=np.zeros((70,24,24))
save_out['activation_3']=np.zeros((60,12,12))


save_out['activation_1_general']=np.zeros((50,80,48,48))


save_out['activation_1_patches']=np.zeros(80)
save_out['activation_2_patches']=np.zeros(70)
save_out['activation_3_patches']=np.zeros(60)


save_out['activation_1_in']=np.zeros((80,45,45))
save_out['activation_2_in']=np.zeros((70,45,45))
save_out['activation_3_in']=np.zeros((60,45,45))
save_out['activation_1_general_in']=np.zeros((50,45,45))

save_imgs=np.zeros((60000,45,45),dtype=np.uint8)



ss=0

for pic_ind,picture in enumerate(get_patches(lookup_test_path,groundtruth_path,patch_size, subsample_mask)):
    if picture[0] is None:
        continue
    ss+=1

    
    if ss%1500==0: print ('making sample for: ',pic_ind,ss)
    rowx,coly=picture[-1]
    #draw_one(picture[0],'pic_({},{})'.format(coly,rowx))
    #draw_one(picture[0][21:-21,21:-21], 'pic_({},{})_small{}'.format(coly, rowx,picture[1] ))
    
    
    save_imgs[ss%60000]=picture[0]
	

# draw_someoutput( save_out['activation_1'],save_picdir,draw_suffix='pic_layer_1'.format())
# draw_someoutput( save_out['activation_1_in'],save_picdir,chosen_cmap='gray',
#                  draw_suffix='pic_layer_1_in'.format())

np.save(r'D:\data_projects\neuron_fibers_data\save_imgs.npy',save_imgs)