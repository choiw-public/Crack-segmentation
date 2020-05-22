# Erratic shaped object segmentation
Deep learning models for segmenting erratic shaped objects. This repository will have more models in future.
* Current project: crack segmentation

### Demo - highlight
* Full demo: https://bit.ly/choiw-fulldemo
* Crack-propagation1:  
![alt text](demo/crack_propagation1.gif)

* Crack-propagation2:  
![alt text](demo/crack_propagation2.gif)

* Crack-moving shot1:  
![alt text](demo/crack_moving_shot1.gif)

* Crack-moving shot2:  
![alt text](demo/crack_moving_shot2.gif)

* Asphalt-moving shot:  
![alt text](demo/asphalt_moving_shot.gif)

### Libraries
* Python v3.x 
* Tensorflow v1.15
* Tensorflow-Probability 0.7.0  
* OpenCV
* Numpy
* Scipy
* Horovod (https://github.com/horovod/horovod)

### Key techniques
* Feature pyramid
* Global attention block
* Bottleneck layer
* Shortcut connection (concatenation and addition)
* Half-precision (FP16)

### Model summary
* Dataset: built from scratch for this side project
* Architecture: depth-42 (38 downscale and 4 upscale), no pretrained model used 
* Number of parameters: 2 million

