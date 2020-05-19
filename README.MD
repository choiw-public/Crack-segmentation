# Segment erratic shaped objects
Deep learning models for segmenting erratic shaped objects  
* Current project: crack segmentation  

#### Demo - highlight
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

#### Libraries
* Python v3.x 
* Tensorflow v1.15
* Tensorflow-Probability 0.7.0  
* OpenCV
* Numpy
* Scipy
* Horovod (https://github.com/horovod/horovod)

#### Key techniques
* Feature pyramid
* Global attention block
* Bottleneck layer
* Shortcut connection (concatenation and addition)
* half-precision (FP16)

#### Model profile
* Number of parameters: 2 million
* Architecture depth: 38 (downscale stream) and 4 (upscale stream)

#### Note for the current state
The current side project (crack segmentation) is finished in my local workstation. The full code will be disclosed in future.
 