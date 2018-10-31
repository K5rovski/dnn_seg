
# DNN_Seg

__Short Description__

This is a Small Machine Learning Framework, for Image Segmentation on Medical Images.

It's build to be modular so that different parts can possibly be applied in neighboring domains.

__So far we have tested these procedures on preprocessed* EM images of:__



1. **_Prefrontal Axon_**


    ![Prefrontal Axon](https://i.imgur.com/VACsO3V.png "Prefrontal Neuron Fiber")

2. **_Optic Nerve Fiber_**


    ![Optic Nerve Fiber](https://i.imgur.com/4RxEa2p.png "Optic Nerve Fiber")




*The images were first preprocessed to 3 separate classes,

 using a clustering tool (BayesNet or KMeans)
______

## Code setup


### Setup Python


#### 1. The Miniconda package was used for python development

 + Download miniconda from [conda](https://conda.io/miniconda.html)
   
#### 2. The file **"setup/dnn_experiment.yaml"** contains the python environment info
   + Inside a terminal, at root execute  **`conda env create -f setup/dnn_experiment.yml`**






* This can be done independently on Linux or other OS, using the yaml environment file, which contains all needed package info...

-------


### Setup ImageJ

ImageJ is an open-source scientific Image Processing Application, with Java Libraries.


1. Download ImageJ in Fiji Form [Fiji](https://fiji.sc/)
2. You can use this module to view images and edit them,
3. And use it for some parts of the pipeline below...


------

## Testing the modules

All the main processing can be viewed using python notebooks:
(also found in the notebooks folder locally)

1. [Sampling Training Instances](https://colab.research.google.com/drive/1P41TTk9QhhklUvlNJlP_uK99XeRz79Yn)
2. [Making a LMDB Instance Base]()
2. [Operating a DNN Instance](https://colab.research.google.com/drive/1ukglFO11jWlIBO7FsnGUSDGwPtNJc7cA)
3. [Operating an AEN Instance]()
3. [Operating a CRBM Instance]()


------

## Running the code

All the code processing is mainly executed using workspaces:

1. `data_prep/sample/sample_set_workspace.py`
2. `data_prep/base/production_script.py`
3. `net/dnn_workspace.py`
4. `net/autoenc_workspace.py`
5.  ~~net/run_crbm.py~~
6. ~~net/utils/_vis_best_maps.py~~

by changing values in the **workspace** files, which then execute separately.

There is a separate code which saves the configuration scripts, for experiment tracking.

The main processing pipeline blocks are described thouroughly in [Neuron Fiber Segmentation](https://docs.google.com/spreadsheets/d/1c5AoThN5RqBoowZb_t5Ak4pKHGL7dD8l-m8MUjJMVp0/edit?usp=sharing).


For any questions, or issues you can email the developer at [Kristijan Petrovski](mailto:petrovski.kristijan@manu.edu.mk).


------


## Ray measuring tool

The ray measuring tool is used to measure the fiber isolation band around a ROI border.


It is build from 4 different processing steps:

1. `raym/run_ray_meas.py`
2. `raym/calc_matches.py`
3.  `raym/find_postc_params.py`
4. `raym/draw_gratio.py`

Also we provide an imagej plugin which automates all the processing, using a single text configuration file.


   **It's located at raym/RayMeasuringPlugin.**

