# Salt-Rock-Detection-using-OpenVINO

I took help fromt the Waldeland's CNN for ASI git repo where he have given his model which is helps us to detect the class of the Rock Bed. This repository contains python/pytorch code for applying Convolutional Neural Networks (CNN) on seismic data. The input is a segy-file containing post-stack seismic amplitude data. The training labels are given as images of the training slices, colored to indicate the classes.

# Dataset
Used the F3 Netherlands dataset - originally made available by OpenDTect - and available via the MalenoV Project.

Downloaded the '.segy'-file and renamed it to 'data.segy' and put it in the 'F3'-folder.

# Using tensorboard (for visualization)
cd to the code-folder
run: tensorboard --logdir='log'
Open a web-browser and go to given pata/URL


# Usage
train.py - train the CNN
test.py - Example of how the trained CNN can be applied to predict salt in a slice or the full cube. In addition it shows how learned attributes can be extracted.
Files
# In addition, it may be useful to have a look on these files

texture_net.py - this is where the network is defined
batch.py - provide functionality to generate training batches with random augmentation
data.py - load/save data sets with segy-format and labeled slices as images
tb_logger.py - connects to the tensorboard functionality
utils.py - some help functions
test_parallel.py - An implemenation of test.py supporting multi-gpu prediction (thanks to Max Kaznady).
Using a different data set and custom training labels

# Changes Made to run using OpenVINO 

I have firstly converted the torch model into Onnx and then used Model Optimizer to generate the IR code.
The obtained .xml and .bin is then passed for Inference using infer.py file.
Then I change the train.py and utils.py according to the inputs obtained form the model given to the infer.py file, and then feed the outputs for further process. 
