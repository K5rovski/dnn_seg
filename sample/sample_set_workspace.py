
from sample_set_funcs import get_good_patches
import os
import glob
import numpy as np
import sys

from sample_set_funcs import ConfigSample
sys.path.append('../..')
from dnn_seg.train.train_helper import save_relevant


# ------------------------------------------------------------------------------------
#
# GENERAL SETUP
#
# ------------------------------------------------------------------------------------


np.random.seed(8)
#np.random.seed(None)


#  Sampling Experiment ID
quick_str='sample_'+''.join(map(chr,np.random.randint(97,97+26,(5,))) )

conf_test_tosave=save_relevant('saved_confs',quick_str,
            files=[ f for f in os.listdir('.') if os.path.isfile(f) and not f.endswith('.pyc')],
            just_return=True)



# ------------------------------------------------------------------------------------
#
# EXPERIMENT SETUP PARAMETERS
#
# ------------------------------------------------------------------------------------

DESCRIPTION_TEXT='sample ONSET, onspecific settings'
save_conf=True


# lookup paths params-------
lookup_path= {}
lookup_path['groundtruth'] = r'D:\data_projects\neuron_fibers_data\images\cors'
lookup_path['interim'] = r'D:\data_projects\neuron_fibers_data\images\intsall'
lookup_path['bigc'] = r'D:\data_projects\neuron_fibers_data\images\bigmask_ON_set_chosen'
lookup_path['smallc'] = r'D:\data_projects\neuron_fibers_data\images\smallmask_ON_set_chosen'
lookup_path['debri'] = '###################'#'../../neuron_fiber_images/sample_imgs/Debris_Interim_1k'

conf=ConfigSample()

# general sample params---
conf.save_loc = r'D:\data_projects\neuron_fibers_data\images\patch_mask_ON_chosenTrainingImgs'
conf.img_size=2400
conf.win_offs_big = 35
conf.win_sparsity_big = 0.12
conf.win_offs_small = 5
conf.win_sparsity_small = 0.15

# noise unif params-------
conf.win_offs_noise = 5
conf.win_sparsity_noise = 0.15
conf.noise_big_sparsity = 0.85
conf.noise_val = 255

# noise roi not used for now-----------------
conf.dbscan_eps = 10
conf.dbscan_mins = 20
conf.win_offs = 22
conf.win_noiseroi_spar = 0.08

# pixel in interims-------------
leave_pix_noise = 0.1
pix_remove_thres = 50


# sampling images params----
train_images=['sp14484-img04', 'sp14485-img05', 'sp13909-img05', 'sp14240-img03', 'sp14069-img04',
 'sp14250-img03', 'sp13750-img09', 'sp13750-img03', 'sp13880-img07',
 'sp14069-img01', 'sp13909-img11', 'sp13909-img07', 'sp14370-img10',
 'sp14240-img01', 'sp14245-img04', 'sp13726-img08', 'sp13880-img11',
 'sp14485-img03', 'sp14485-img09', 'sp14370-img07']
	
	
	

# =============================================================================
# =============================================================================


# ------------------------------------------------------------------------------------
#
# RUNNING THE SETUP CONFIGURATION
#
# ------------------------------------------------------------------------------------



if not os.path.exists(conf.save_loc):
    os.makedirs(conf.save_loc)



test_images = {}
for pic_type in 'groundtruth;interim;bigc;smallc'.split(';'):
    print glob.glob(os.path.join(lookup_path[pic_type], '*.tif'))
    test_images[pic_type] = [filename for ind, filename in
                             enumerate(glob.glob(os.path.join(lookup_path[pic_type], '*.tif')))
                             if filename[filename.rindex('sp'):filename.rindex('img') + 5] \
                             in train_images]
    test_images[pic_type].sort(key=lambda name: name[name.rindex('sp'):])
 # print test_images[pic_type][:3]


print len(test_images['groundtruth']), test_images['groundtruth'],

get_good_patches(test_images, conf.save_loc, conf.win_offs,conf)


if save_conf:
    save_relevant('saved_confs',quick_str,str_to_save=conf_test_tosave,descriptive_text=DESCRIPTION_TEXT)