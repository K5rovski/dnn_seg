import pickle
from keras.callbacks import Callback
from os.path import join


class LogLossesCallback(Callback):

    def __init__(self,num_batches_validate,save_freq,model_id='autoenc',save_loc='.',do_single_patch=False,save_model=None,val_data=None):
        self.batch_val=num_batches_validate
        self.val_data=val_data
        self.model_id=model_id
        self.save_loc=save_loc
        self.savemodel_loc=save_model
        self.save_freq=save_freq
        self.do_single_patch=do_single_patch

        if do_single_patch:
            pass


        super(LogLossesCallback,self).__init__()

    def on_train_begin(self, logs={}):


        self.losses = []
        self.val_losses=[]


    def on_batch_end(self, batch, logs={}):
        self.losses.append(( float(logs.get('loss')),float(logs.get('acc',-1)) ))

        if batch%self.batch_val==0:
            pass
            score=self.model.evaluate(self.val_data[0],
                                self.val_data[1],batch_size=300,verbose=0)
            pass
            self.val_losses.append(score)
        if batch%(self.save_freq)==0 and self.savemodel_loc is not None:
            self.model.save(join(self.savemodel_loc,
                'model_inside_{}_runid{}.hdf5'.format(self.model_id, batch//(self.save_freq)  )))

    def on_epoch_end(self,epoch,logs={}):
        with open(join(self.save_loc, 'history_model_ep{}_{}.hist'.format(epoch,self.model_id)), 'wb') as f:
            pickle.dump((self.losses,(self.batch_val,self.val_losses),self.save_freq), f)