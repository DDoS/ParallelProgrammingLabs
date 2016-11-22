# Report

Aleksi Sapon-Cousineau - 260581670  
Samuel Cauchon - 260587509

## 0. Parallelization

Describe what `findBestGridAndBlockDims2D` in `common.h` does here.

## 1. Image transformations

### 1.0. CUDA implementation

Describe use of texture object API in `transform.cu` here. Advantages: better cache usage, better modelling of the input data since designed to work with images.

### 1.1. Performance analysis

Two different GPUs. More streaming multiprocessors makes it faster, compare the number for each model.

Time measured with CUDA event timing.

### 1.2. Rectification

|NVIDIA GPU     |Time (ms)|
|---------------|---------|
|GeForce GTX 680|0.0803   |
|Tesla K40c     |0.0524   |

Talk about complexity, 1:1 mapping from input image to output.

![Input image](Rooster.png)  
The input image used for the rectification performance tests.

![Output image](RoosterRectified.png)  
The output image from the rectification performance tests.

### 1.3. Pooling

|NVIDIA GPU     |Time (ms)|
|---------------|---------|
|GeForce GTX 680|0.0215   |
|Tesla K40c     |0.0289   |

Talk about complexity, 4 input pixels per output pixel, but output image is 4 times smaller.

![Input image](Jaguar.png)  
The input image used for the pooling performance tests.

![Output image](JaguarPooled.png)  
The output image from the pooling performance tests.

### 1.4. Convolution

|NVIDIA GPU     |Time (ms)     |
|---------------|--------------|
|GeForce GTX 680|.......       |
|Tesla K40c     |.......       |

Talk about complexity, 9 input pixels per output pixel, and output image is only marginally smaller.

![Input image](JustDoIt.png)  
The input image used for the convolution performance tests.

![Output image](JustDoItConvolved.png)  
The output image from the convolution performance tests.

## 2. Finite element synthesis

### 2.0. CUDA implementation

Describe use of surface object API in `grid.cu` here. Advantages: makes read and write addressing easier, better modelling of simulation data because it is designed for multidimensional arrays.

### 2.1. Performance analysis

Two different GPUs. More streaming multiprocessors makes it faster, compare the number for each model.

Time measured with bash `time` command.

### 2.2. 4 by 4 grid

|NVIDIA GPU     |Time (seconds)|
|---------------|--------------|
|GeForce GTX 680|0.314         |
|Tesla K40c     |0.383         |

Very small simulation, how does that affect the performance difference.

### 2.2. 512 by 512 grid

|NVIDIA GPU     |Time (seconds)|
|---------------|--------------|
|GeForce GTX 680|1.039         |
|Tesla K40c     |0.964         |

Larger simulation, how does that affect the performance difference.
