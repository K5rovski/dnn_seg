from keras.engine import Layer
from keras.models import load_model
import tensorflow as tf
import os
import sys
import re

sys.path.append('../../..')


from dnn_seg.net.utils.train_h \
        import make_intmodel


class AutoEncoderLayer(Layer):


    def __init__(self,conf_dikt,default_model_loc=None,**kwargs):
        self.conf_dikt=conf_dikt
        self.def_autoenc=default_model_loc

        super(AutoEncoderLayer,self).__init__(**kwargs)


    def build(self,input_shape):
        autoenc_loc = self.conf_dikt['autoenc_loc']
        if  'replication' in autoenc_loc: 
            autoenc_loc=re.sub(r'\\','/',autoenc_loc)
            autoenc_last_loc=autoenc_loc.split(r'/')[-1]
            autoenc_loc=os.path.join(AutoEncoderLayer.default_autoenc_loc(),
                autoenc_last_loc)
        # print(__file__)
        auto_enc = load_model(autoenc_loc, custom_objects={'tf': tf})
        chosen_lay, chosen_lname = [(lay, lay.name) for lay in auto_enc.layers \
                                    if lay.name == self.conf_dikt['autoenc_layer']][0]
        self.weightM,self.hbias=chosen_lay.get_weights()


    def call(self,x,mask=None):
        added_vbias = x

        conved = tf.nn.conv2d(added_vbias
                              , self.weightM, padding='SAME', strides=(1, 1, 1, 1))
        added_hbias=conved+self.hbias
        return added_hbias

    def get_config(self):

        base_config = super(AutoEncoderLayer, self).get_config()
        base_config['conf_dikt']=self.conf_dikt
        return base_config

    def get_output_shape_for(self, input_shape):
        return input_shape[:3]+(self.weightM.shape[-1],)


    @classmethod
    def default_autoenc_loc(cls):
        return '../test_data'
