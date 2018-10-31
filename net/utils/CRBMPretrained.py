from keras import backend as K
from keras.engine.topology import Layer
import numpy as np
import tensorflow as tf

def get_pretrained(model_loc):
    w_found = np.load(model_loc.format('Weight'))
    vbias = np.load(model_loc.format('vbias'))
    hbias = np.load(model_loc.format('hbias'))
    return w_found, vbias, hbias


class CRBMPretrained(Layer):

    def __init__(self, output_dim,path_pretrain, **kwargs):
        self.output_dim = output_dim
        self.path_pretrain=path_pretrain
        super(CRBMPretrained, self).__init__(**kwargs)

    def build(self, input_shape):
        if self.path_pretrain is None:
            raise
        self.weightM,self.vbias,self.hbias=get_pretrained(self.path_pretrain)

        super(CRBMPretrained, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x,mask=None):
        added_vbias=x#+self.vbias
        # conved=K.conv2d(added_vbias, self.weightM, strides=(1,1),
        #                   border_mode='valid',dim_ordering='tf'
        #                   )

        conved=tf.nn.conv2d(added_vbias
                            ,self.weightM,padding='VALID',strides=(1,1,1,1))
        added_hbias=conved+self.hbias
        return added_hbias


    def get_config(self):
        config = {'path_pretrain':self.path_pretrain,
                  'output_dim':self.output_dim
                  }
        base_config = super(CRBMPretrained, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def get_output_shape_for(self, input_shape):
        return (input_shape[0],)+ tuple(self.output_dim)