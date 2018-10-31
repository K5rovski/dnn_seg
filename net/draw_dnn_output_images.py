from keras.models import load_model
import sys




sys.path.append('../..')

from dnn_seg.net.utils import AutoEncoderLayer
from dnn_seg.net.utils.train_h import quick_test





big_img_size=2048

test_parDikt={
    'patch_size':45,
    'img_path':r'D:\data_projects\neuron_fibers_data\TCINT_neigh15_circ10\\',
    'groundtruth':r'D:\data_projects\neuron_fibers_data\cors_mlset01\\',
    'interim':r'D:\data_projects\neuron_fibers_data\TCINT_neigh15_circ10\\',

	'prediction_saveloc':r'D:\data_projects\neuron_fibers_data\predictions',
		'DNNOutput_saveloc':r'D:\data_projects\neuron_fibers_data\dnnoutput',
		}







do_whole_pic=True
draw_img_ext='*.tif'
split_patches=True





print('Drawing DNNOutput Images...')
save_model_loc = r'D:\data_projects\neuron_fibers_data\models\dnnmodel_hemv.00-0.9992.hdf5'

model=load_model(save_model_loc,custom_objects={'AutoEncoderLayer':AutoEncoderLayer.AutoEncoderLayer})


quick_oldstr=save_model_loc[save_model_loc.rindex('dnnmodel_')+8:save_model_loc.rindex('hdf5')]
quick_oldstr+='mlset01'


for older,newer,xx in zip(range(0,10,2),range(1,15,2),range(1)):

	test_parDikt.update({'from': older, 'to': newer})

	quick_test(model, quick_oldstr, big_img_size, do_all=do_whole_pic,
			   draw_img_ext=draw_img_ext, test_batch_size=500,
			   split_map=split_patches,
			   conf_dikt=test_parDikt,two_patch_instance=True)
print('I\'m done drawing dnnoutputs')