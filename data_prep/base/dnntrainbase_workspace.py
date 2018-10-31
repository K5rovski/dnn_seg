import glob
import os
import time
import numpy as np
import sys


sys.path.append('../../..')
from dnn_seg.data_prep.utils.create_lmdb_batch_funcs import reduce_files,make_db_folder,create_lmdbs
from dnn_seg.net.utils.train_h import save_relevant


#SETUP TO SAVE THIS WORKSPACE EXPERIMENT FILE    ============================


np.random.seed(None)
quick_str='dnnbase_'+''.join(map(chr,np.random.randint(97,97+26,(5,))) )

print('If run successfully, this script is saved as {}'\
	.format(os.path.join('saved_baseconfs',quick_str) ) )



files_save=[]
files_save.insert(0,os.path.basename(__file__))

conf_test_tosave=save_relevant('saved_baseconfs',quick_str,
            files=files_save,
            just_return=True)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>





DESCRIPTION_TEXT='normal sample, interim ON, 6 mil, specific images and sampling, 45window '

db_folder=r'D:\data_projects\neuron_fibers_data\BASES\8mil_61_bigp_Wsize_20imgs_interimONset'

save_base=True
image_width,image_height=45,45
big_image_size=2048
lookup_path = r'D:\data_projects\neuron_fibers_data\images\ML_SET_ON01\cints_new_general'

groundtruth_path = r'D:\data_projects\neuron_fibers_data\images\cors\c'



val_images='sp14252-img06,sp14436-img05,sp14436-img04,sp14252-img08'.split(',')



test_image_val=[filename for ind,filename in enumerate(glob.glob(os.path.join(lookup_path, '*.tif')) ) \
if  filename[filename.rindex(os.path.sep)+1:filename.rindex('img')+5] in val_images ]

val_do_expand=False
percent_patches_used_val=np.array([1,1,1])*0.1  
phase_done_val='val'
val_batch_size=5*10**5
val_lmdb_GBsize=20



train_mask_dir=r'D:\data_projects\neuron_fibers_data\images\patch_mask_ON_chosenTrainingImgs\\'


train_images=('sp14436-img08,sp13909-img01,sp13909-img02,sp13938-img07,sp13726-img01,'+
'sp14252-img05,sp13909-img07,sp13909-img08,sp13909-img03,sp13933-img04,sp13726-img04,'+
'sp13726-img03,sp13933-img08,sp13909-img05,sp13726-img02,sp13933-img05,sp13933-img06,'+
'sp13933-img07,sp13909-img06,sp13726-img08').split(',')




test_image_train=[filename for ind,filename in enumerate(glob.glob(os.path.join(lookup_path, '*.tif')) )\

 if  filename[filename.rindex(os.path.sep)+1:filename.rindex('img')+5] in train_images ]

train_do_expand=False
percent_patches_used_train= np.array([1,1,1])   #*(1.0/8)  #[4.0/19,7.0/10,1]
phase_done_train='train'

train_batch_size=5*10**5
train_lmdb_GBsize=60

do_special={'perc_switch':0,'not_mix_patches':True,
            'two_patch':True,'folded_expand':True}








# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

start_time = time.time()
print ('Creating images at "%s" ...' % db_folder)
print( 'Working on: ',len(test_image_val),test_image_val)

make_db_folder(db_folder)

create_lmdbs(db_folder,    phase_done_val,    (test_image_val,lookup_path, groundtruth_path, "all_test", 0, '###'),'###','###',image_width,image_height,smaller_size=percent_patches_used_val,random_key_prepend=12,do_expand=val_do_expand,patch_size=image_width,save_base=save_base,batch_size=val_batch_size,base_GBsize=val_lmdb_GBsize,
            do_special=do_special
            )
reduce_files(db_folder,phase_done_val)

print( 'Done after {:.2f} hours'.format((time.time() - start_time)/3600 ))


start_time = time.time()

print ('Working on: ',len(test_image_train),test_image_train)

create_lmdbs(db_folder,    phase_done_train,    (test_image_train,lookup_path, groundtruth_path, "all_test", 0, '###'),'###','###',image_width,image_height,smaller_size=percent_patches_used_train,mean_name='mean.jpg',random_key_prepend=12,do_expand=train_do_expand,patch_size=image_width,save_base=save_base,batch_size=train_batch_size,mask_dir=train_mask_dir,base_GBsize=train_lmdb_GBsize,
           do_special=do_special
            )
reduce_files(db_folder,phase_done_train)

print( 'Done after {:.2f} hours'.format((time.time() - start_time)/3600 ))



# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

save_relevant('saved_baseconfs',quick_str,str_to_save=conf_test_tosave,descriptive_text=DESCRIPTION_TEXT)