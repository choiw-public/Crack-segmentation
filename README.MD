# Crack Segmentation
A deep learning model for segmenting cracks. This repository will have more models in future.

### Demo
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
* TensorFlow v1.15
* TensorFlow-Probability 0.7.0
* Horovod (https://github.com/horovod/horovod)  
* OpenCV
* NumPy
* SciPy


### Key techniques
* Half-precision (FP16)
* Feature pyramid
* Global context block
* Bottleneck layer
* Shortcut connection (concatenation and addition)


### Model summary
* Dataset: built from scratch for this side project
* Architecture: depth-42 (38 downscale and 4 upscale), no pretrained model used 
* Number of parameters: 2 million

