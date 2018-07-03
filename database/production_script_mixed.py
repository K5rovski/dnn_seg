import glob
import os
import time
import pickle
import numpy as np
import sys
sys.path.append('../..')



#SETUPP    ============================
from data_funcs.create_lmdb_batch_funcs import reduce_files,make_db_folder,create_lmdbs
from data_funcs.data_helper import create_chessmask

from dnn_seg.train.net_models import get_2k_image_2layer_convnetmodel
from dnn_seg.train.train_helper import save_relevant

from keras.optimizers import SGD

# ----------------------------------------------------------------------


image_list_folder='#########'
image_prepend='###########'
save_path = '#################'
patch_dir='##################'
save_path = '#################'
# =============================================



np.random.seed(None)
quick_str='mixbase_'+''.join(map(chr,np.random.randint(97,97+26,(5,))) )
files_save=[ os.path.join('data_funcs',f) for f in os.listdir('data_funcs') \
             if os.path.isfile(os.path.join('data_funcs',f))
             and not os.path.join('data_funcs',f).endswith('.pyc')]
files_save.insert(0,os.path.basename(__file__))

conf_test_tosave=save_relevant('data_funcs/saved_mixbaseconfs',quick_str,
            files=files_save,
            just_return=True)

# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

DESCRIPTION_TEXT='6 milion normal older sample,20 images train,80 percent noise mix patches,autoencoder patches, 45 window'

db_folder='D:/NEURON_FIBER_DATA/lmdb_base_2k_45w_autoenc_normal_mask_6milset_mixedset_80perc_20pics_prod'

save_base=True
image_width,image_height=45,45
big_image_size=2048
lookup_path = r'D:\code_projects\dnn_seg\workspace\Sampling\Saved_Noises_oldsample_prod\\' #"D:/code_projects/Neuron_fibers_workspace/neuron_fiber_images/all_test_big_spread_interim/"

groundtruth_path =r'D:\code_projects\dnn_seg\data\Consensus corrected\\'



val_images='sp14252-img06,sp14436-img05,sp14436-img04,sp14252-img08'.split(',')
test_image_val=[filename for ind,filename in enumerate(glob.glob(os.path.join(lookup_path, '*.tif')) ) \
if  filename[filename.rindex(os.path.sep)+1:filename.rindex('img')+5] in val_images ]

val_do_expand=False
percent_patches_used_val=np.array([1,1,1])*0.1  #np.array([0.837,0.744,1])*0.1 #[4.0/19,7.0/10,1]
phase_done_val='val'
val_batch_size=5*10**5
val_lmdb_GBsize=30



train_mask_dir=r"D:\code_projects\dnn_seg\workspace\Sampling\patch_mask_cells_prod_fullset_oldsample_mixed\\"

contested_img='sp13726-img07'
train_images='sp14436-img08,sp13909-img01,sp13909-img02,sp13938-img07,sp13726-img01,sp14252-img05,sp13909-img07,sp13909-img08,sp13909-img03,sp13933-img04,sp13726-img04,sp13726-img03,sp13933-img08,sp13909-img05,sp13726-img02,sp13933-img05,sp13933-img06,sp13933-img07,sp13909-img06,sp13726-img08'.split(',')

test_image_train=[filename for ind,filename in enumerate(glob.glob(os.path.join(lookup_path, '*.tif')) )\

 if  filename[filename.rindex(os.path.sep)+1:filename.rindex('img')+5] in train_images ]

train_do_expand=False
percent_patches_used_train= np.array([1,1,1])#*(1.0/8)  #[4.0/19,7.0/10,1]
phase_done_train='train'

train_batch_size=5*10**5
train_lmdb_GBsize=60

do_special={'perc_switch':0.8,'two_patch':True}

print(test_image_train)

# ---------------------------------------------------------------------------------------------------

# TRAINING IN KERAS

# MODEL_NAME: TEST_INTERIM_2K_45W_ROI_CELLSNOISE_BIGMAP_NONEXP_BRAIN_1M_PETRINET

# SAVED MODEL: TEST_INTERIM_2K_45W_ROI_CELLSNOISE_BIGMAP_NONEXP_BRAIN_1M_PETRINET
# ---------------------------------------------------------------------

with open(r'D:\code_projects\Neuron_fibers_workspace\models\TEST5_INTERIM_2K_45W_ROIUNIF_BIGMAP_NONEXP_BRAIN_1M_PETRINET_11epoch\weights.all','rb') as f:
    weights_all=pickle.load(f)



deploy_batch_size=180

lookup_test_path = "D:/code_projects/Neuron_fibers_workspace/neuron_fiber_images/test_spread_interim/"

model_loc= r'D:\code_projects\Neuron_fibers_workspace\models\replication\weights_ada_finetune.14-0.86.hdf5'

# model=get_2k_image_2layer_convnetmodel(45,45)
# model.compile(loss='categorical_crossentropy',
#               optimizer= SGD(lr=0.00008, decay=0.0005, momentum=0.9, nesterov=False),
#               metrics=['accuracy'])
#
# model.load_weights(model_loc,by_name=True)

output_prediction_file="D:/code_projects/Neuron_fibers_workspace/prediction_files/just_architecture_testing_replicate_adaontop.txt"

colored_images='D:/code_projects/Neuron_fibers_workspace/neuron_fiber_images/visualised_predictions/just_architecture_testing_replicate_adaontop/'
colors_colored=(255,0,0),(0,0,255),(0,255,0),
colors_grey=(0,0,0),(128,128,128),(255,255,255)
color_mistake=(25,255,37)


test_image_paths=[filename for ind,filename in enumerate(glob.glob(os.path.join(lookup_test_path, '*.tiff')) ) \
if  filename[filename.rindex(os.path.sep)+1:filename.rindex('img')+5] ]







# -----------------------


# =====================================================================
start_time = time.time()
print ('Creating images at "%s" ...' % db_folder)
print( 'Working on: ',len(test_image_val),test_image_val)

make_db_folder(db_folder)

# create_lmdbs(db_folder,    phase_done_val,    (test_image_val,lookup_path, groundtruth_path, "all_test", 0, save_path),image_list_folder,image_prepend,image_width,image_height,smaller_size=percent_patches_used_val,random_key_prepend=12,do_expand=val_do_expand,patch_size=image_width,save_base=save_base,batch_size=val_batch_size,base_GBsize=val_lmdb_GBsize,
#             )
reduce_files(db_folder,phase_done_val)

print( 'Done after {:.2f} hours'.format((time.time() - start_time)/3600 ))


start_time = time.time()

print ('Working on: ',len(test_image_train),test_image_train)

create_lmdbs(db_folder,    phase_done_train,    (test_image_train,lookup_path, groundtruth_path, "all_test", 0, save_path),image_list_folder,image_prepend,image_width,image_height,smaller_size=percent_patches_used_train,mean_name='mean.jpg',random_key_prepend=12,do_expand=train_do_expand,patch_size=image_width,save_base=save_base,batch_size=train_batch_size,mask_dir=train_mask_dir,base_GBsize=train_lmdb_GBsize,
           do_special=do_special
            )
reduce_files(db_folder,phase_done_train)

print( 'Done after {:.2f} hours'.format((time.time() - start_time)/3600 ))

# ====================================================================
# !!!!!!!!!!! TEST_PHASE

# ---
# ---




# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
# >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>

save_relevant('data_funcs/saved_mixbaseconfs',quick_str,str_to_save=conf_test_tosave,descriptive_text=DESCRIPTION_TEXT)

