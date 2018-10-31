
import os
import glob
import numpy as np
import sys

sys.path.append('../../..')
from dnn_seg.net.utils.train_h import save_relevant
from dnn_seg.data_prep.utils.sample_set_funcs import get_good_patches,ConfigSample


#SETUP TO SAVE THIS WORKSPACE EXPERIMENT FILE    ============================



np.random.seed(8)


#  Sampling Experiment ID
quick_str='sample_'+''.join(map(chr,np.random.randint(97,97+26,(5,))) )

print('If run successfully, this script is saved as {}'\
	.format(os.path.join('./saved_sampleconfs',quick_str) ) )

conf_test_tosave=save_relevant('saved_sampleconfs',quick_str,
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
lookup_path['groundtruth'] = r'D:\data_projects\neuron_fibers_data\cors_mlset01_31'
lookup_path['interim'] = r'D:\data_projects\neuron_fibers_data\TCINT_neigh15_circ10_31'
lookup_path['bigc'] = r'D:\code_projects\Neuron_fibers_workspace\neuron_fiber_images\sample_imgs\NEWINTERIM_bigrois'
lookup_path['smallc'] = r'D:\code_projects\Neuron_fibers_workspace\neuron_fiber_images\sample_imgs\NEWINTERIM_smallrois'
lookup_path['debri'] = '###################' # not used, for now

conf=ConfigSample()

# general sample params---
conf.save_loc = r'D:\data_projects\neuron_fibers_data\images\patch_mask_chosenTrainingImgs_test'
conf.img_size=2048
conf.win_offs_big = 22
conf.win_sparsity_big = 0.12
conf.win_offs_small = 12
conf.win_sparsity_small = 0.12

# noise unif params-------
conf.win_offs_noise = 5
conf.win_sparsity_noise = 0.15
conf.noise_big_sparsity = 0.9
conf.noise_val = 200

# noise roi not used for now-----------------
conf.dbscan_eps = 10
conf.dbscan_mins = 20
conf.win_offs = 22
conf.win_noiseroi_spar = 0.08

# pixel in interims-------------
leave_pix_noise = 0.1
pix_remove_thres = 50


# sampling images params----
	
train_images =( 'sp13726-img07,sp13909-img01,sp13909-img02,sp13938-img07,'+
'sp13726-img01,sp14252-img05,sp13909-img07,sp13909-img08,sp13909-img03,'+
'sp13933-img04,sp13726-img04,sp13726-img03,sp13933-img08,sp13909-img05,'+
'sp13726-img02,sp13933-img05,sp13933-img06,sp13933-img07,sp13909-img06,sp13726-img08').split(',')


train_images = 'sp13726-img07,sp13909-img01,sp13909-img02'.split(',')



# ------------------------------------------------------------------------------------
#
# RUNNING THE SETUP CONFIGURATION
#
# ------------------------------------------------------------------------------------



if not os.path.exists(conf.save_loc):
    os.makedirs(conf.save_loc)



test_images = {}
for pic_type in 'groundtruth;interim;bigc;smallc'.split(';'):
 
    test_images[pic_type] = [filename for ind, filename in
                             enumerate(glob.glob(os.path.join(lookup_path[pic_type], '*.tif')))
                             if filename[filename.rindex('sp'):filename.rindex('img') + 5] \
                             in train_images]
    test_images[pic_type].sort(key=lambda name: name[name.rindex('sp'):])

	

print('Sampling {} images = \n{}'.format(len(test_images['groundtruth']), test_images['groundtruth'],))

get_good_patches(test_images, conf.save_loc, conf.win_offs,conf)


if save_conf:
    save_relevant('saved_sampleconfs',quick_str,str_to_save=conf_test_tosave,descriptive_text=DESCRIPTION_TEXT)