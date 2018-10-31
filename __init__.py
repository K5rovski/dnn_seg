'''
# FrontalLobe_DNN_Seg

Main package for DNN Segmentation on Neuron Fibers.

Containing 3 main modules

1. Data Preparation
   + Choosing Sampling Instances
   + Saving Training Instances in a lmdb base, for convinience.
2. Neural Networks Processing
   + Pretraining Module (Autoencoder or Convolutional RBM network) 
   + Deep Neural Network (Using to DNN correct pre-processed images.)
3. Ray Measuring Neuron fiber ROIs
   + Ray Measuring ROIs
   + Matching ROIs between two sets (Annotation, DNN Corrected)
   + Discarding some DNN Damaged ROIs
'''

pass

# __all__=''.split(',')