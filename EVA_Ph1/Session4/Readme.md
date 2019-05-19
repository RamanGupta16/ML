**Session 4 Results Summary**
* 1st DNN: 99.07% validation accuracy with 20,922 parameters. Plain Vanilla network.
* 2nd DNN: 99.26% validation accuracy with 19,042 parameters. Increased BatchSize and number of Epochs.
* 3rd DNN: 99.53% validation accuracy with 14,692 parameters. Added BatchNormalization and Dropout. Best Validation Accuracy.
* 4th DNN: 99.43% validation accuracy with 9,424 parameters. Added LR scheduler. Most efficient in terms of parameters.

**Archiectural Basics**
The following architectural basics are building blocks to define DNN architecture. They are applied in order to arrive at network architecture. The network architecture is not an eaxct science but an art which comes by trial-error and experience.

1. **How many layers**: 
  * One input layer
  * Multiple hidden layer: It depends upon:-
    - Input image complexity and class variations within input.
    - Receptive Field required.
    - Hardware for training and deployment.
    - Minimum acceptable trainable parameters and hyper parameters.
    - Network Architecture.
  * One Output Layer

2. **Receptive Field (RF)**
   * In CNN neuron is connected only to a local region of the previous layer which is defined as its receptive field. 
     Example for a 3x3 convolution the Receptive Field is 3x3 = 9 neurons of previous layer or in case of input image 
     it is 3x3 =9 pixels.
   * The *local* receptive field is same as size of kernel, but the *global* receptive field is the cumulative 
     receptive field seen so far. Thus adding a convolution layer adds to the *global* receptive field.
   * Final global RF may not cover entire image. But it must cover the portion of image required for detection. RF plays
     an important role to determine number of layers and parameters in the network architecture.

3. **3x3 Convolutions**
  * Standard filter/kernel in CNN.
  * The depth or channels of this filter is always equal to that of input image. 
  * Applying 3x3 filter over image gives output (Stride = 1, No Padding):
    - 28x28x3   	3x3x3, 16 		==> 		26x26x16   where 16 is number of filters
    - 26x26x16  	3x3x16, 32 		==> 		24x24x32  where 32 is number of filters

4. **MaxPooling**:
  * Reduces channel resolution and double Receptive Field
  * Applied after minumum 2-3 convolution layers.
  * MaxPool Layer is 2-3 layers away from ouput layer
  
5. **1x1 Convolutions**
  * 1x1 convolution is also called Pointwise convolution. This is a special filter used in CNN. 
  * This filter changes the channel/depth dimension of the input filter image without changing other two dimensions.
  * 1x1 mixes and merges all the channels of input image to produces a single value, for required number of channels. 
 
6. **SoftMax**
  * Used in output layer to highlight the most dominant value in the given vector.
  * The sum of values of softmax output is always 1. It is probability like, but not probability. 

7. **Learning Rate**
   * Learning Rate is a hyper parameter which controls the change made to weights during gradient descent of back propogation.
   * It defines the rate of change made to parameters to reach the minima of loss function.
   * If LR is too small then the training will be slow and it takes large time to train the network .
   * If it is too large then we may never reach or skip the minima of loss function thus never converge.
   * It must be optimal and typically in range of [0.001 - 0.0001]. 
   * It can be changed dynamically per epoch using LR scheduler.
   
8. **Kernels and how do we decide the number of kernels?**
   * Kernel or feature extractors convolve over the image to extract features.
   * The number of kernel depends upon:-
     - Complexity of images and class variations within input.
     - No. of features required to be extracted.
     - Required edges/gradients, textures, patterns, parts-of-object and object for image detection.
   
9. **Batch Normalization**
   * Normalization of channels to bring them in range [-1 to 1 ]
   * Bring all channels within same amplitude range.
   * Batch Normalization improves validation accuracy.
   * It can be applied after every convolution layer.
   * It is not applied at output layer, but before output layer.
   
10. **Image Normalization**
   *

11. **Position of MaxPooling**
   * Applied after minumum 2-3 convolution layers.
   * MaxPool Layer is 2-3 layers away from ouput layer.
   * It is present in Transition Block

12. **Concept of Transition Layers**
   * It has 1x1 and MaxPooling layer
   * It reduces the channel size and number of channels from the previous Convolution Block.
   
13. **Position of Transition Layer**
   * It is defined between two Convolution Blocks.
  
14. **Number of Epochs and when to increase them**
   * No. of epochs is increased to train more on training data.
   * When the training loss and validation loss stop to improve, then we stabilize the no. of epochs.
   
15. **DropOut**
   * Droput is a regularization technique to reduces overfitting of training data.
   * It reduces gap between between Test Accuracy and Training Accuracy.
   * It can be applied after every convolution layer.
   * It is not applied at output layer, but before output layer.
   * Typical values of Droput is in range 0.05 - 0.15

16. **When do we introduce DropOut, or when do we know we have some overfitting**
   * Overfitting is when trained parameters become very specific to training data.
   * Overfitting reduces generalization of DNN and it performs poorly on validation set.
   * When gap between Test Accuracy and Training Accuracy is high it indicates Overfitting.
   * Dropout is used in such cases to reduce overfitting and make DNN more generalized.
   
17. **The distance of MaxPooling from Prediction**
   * MaxPool Layer is 2-3 layers away from ouput layer.
   * It cannot be very near to Prediction Layer because at that layer we want linear relation.
   
18. **The distance of Batch Normalization from Prediction**
   * Batch Normalization is applied before the Prdiction Layer and not at the Prediction layer.
  
19. **When do we stop convolutions and go ahead with a larger kernel or some other alternative (which we have not yet covered)**
   * If input image is large (say 100x100) and we are at later stage where channel resolution is say 9x9 then we can directly 
     use 9x9 kernel instead of continuing with multiple 3x3, before arriving at output layer.
   * This reduces no. of channels and help quickly reach output layer with less no. of hidden layers.
   
20. **How do we know our network is not going well, comparatively, very early**
   * If in initial few layers (2-3) the training loss is high, training acuuracy is low then our network is not doing well.
   * If initially validation loss is high and validation accuracy then our network is not doing well.
   * If gap between training and validation accuracy does not come down in initial few layers then then our network is not doing well.
   
21. **Batch Size, and effects of batch size**
   * Batch Size define how many images from an epoch can be trained in parallel.
   * Lower batch (10-60) size will increase time required to train an epoch completely.
   * Generally higher batch size (64-512) will improve training/validation accuracy since more images are trained in parallel
     and back propogation will have wider dataset to nudge the parameters.
   * Batch Size also depends upon training image dataset complexity. Complex images with rich channels may require lower batch size.
   * Higher batch size can also train quickly because it takes more images in one batch and less time to ccompete 1 epoch.
   
22. **When to add validation checks**
   * Validation check should always be added while training to get the best possible set of trained parameters and validation 
     accuracy in each epoch.
   
23. **LR schedule and concept behind it**
   * Learning Rate is a hyper parameter which controls the change made to weights during gradient descent of back propogation.
   * LR schedule systematically reduces the LR for every epoch according to a cyclic function.
   * High LR will not allow gradient descent to converge and very low LR can take very latrge time to train.

24. **Adam vs SGD**
   * These are common Gradient descent optimization algorithms.
   * These algorithms minimize the loss function by finding its minima. 
   
25. **Hardware**
   * The design of network, choice of building blocks depends very much upon targeted hardware for training and deployment.
   * A low end hardware with limited GPU and memory will force use of less parameters, less accuracy, use of strides for convolution etc.
   * For Example a system which does not require exact iamge detection or does not require high level of accuracy may settle for simple network with less accuracy.
   
