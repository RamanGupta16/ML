Q1. What are Channels and Kernels (according to EVA)?


A1. Kernel or Filter is a small learnable weight matrix (width, height) which extends through the full depth of input.
    Kernel is used to extract features from the input image into a channel or feature map. We compute dot product of
    Kernel weights and input and sum it optionally with bias, followed by activation function to produce a 2-dimensional
    feature map or channel. Kernel/Filter represent learnable features like edges & gradients, texture, patterns,
    parts of obbject and object. 


====================================================================================


Q2. Why should we only (well mostly) use 3x3 Kernels?

A2. Kernel is a learnable weight matrix typically of 3x3 dimension. A 3x3 kernel is used because:

* It reduces number of parameters than a higher dimension kernel. Multiple 3x3 kernels applied in succession have less
  parameters to store than lesser number but higher dimension kernels applied, to arrive at same channel.
* Many GPU vendors provide hardware acceleration for 3x3 kernel thus increasing performance.
* 3x3 kernel provides symmetry axis.


======================================================================================


Q3. How many times do we need to perform 3x3 convolution operation to reach 1x1 from 199x199 (show calculations)

A3. 
