# Report

Aleksi Sapon-Cousineau - 260581670  
Samuel Cauchon - 260587509

## 0. Parallelization

In both labs, the parallelization strategy is to split the problem (the image for lab 1 or the drum for lab 2) into sub problems (blocks) where each subproblems would be solved by a unit (processes and threads in the previous lab). Here, we used CUDA which uses the graphic card to process the data. In order to make both labs paralellized using CUDA, we created a function called `findBestGridAndBlockDims2D` in `common.h`. The goal of this function is to find the best block partition given a 2D array. Using those partitions, we managed to split each of our subproblems and process it on each thread block of our graphic card.

## 1. Image transformations

### 1.0. CUDA implementation

The CUDA implementation we designed for this lab also included a texture object in `transform.cu`. There area few advantages of using this type of object. First of all, we make a better use of the cache, which in turn makes our program more efficient and faster. The second advantage is that it creates a better modeling of the input data since it was designed to work with images.

### 1.1. Performance analysis

For the timing analysis of each type of conversion, we used CUDA event timming. This insured to only get the time that the GPU utilized to process the data. Furthermore, we also tested each conversion using 2 different GPUs to show the performance of both and draw a conclusion on the performance of a CUDA paralellized program and the graphic card used to run the program.

### 1.2. Rectification

|NVIDIA GPU     |Time (ms)|
|---------------|---------|
|GeForce GTX 680|0.0803   |
|Tesla K40c     |0.0524   |

Here we notice that the Tesla K40c graphic card performs better than the GeForce GTX 680 due to the fact that the K40c took 34.74% less time to run the program than the GTX 680. Since it is a 1:1 mapping where 1 value of the input is process to ouput 1 value, the complexity of the program is lowered and this makes it a good program to test each GPU in an efficient way.

![Input image](Rooster.png)  
The input image used for the rectification performance tests.

![Output image](RoosterRectified.png)  
The output image from the rectification performance tests.

### 1.3. Pooling

|NVIDIA GPU     |Time (ms)|
|---------------|---------|
|GeForce GTX 680|0.0215   |
|Tesla K40c     |0.0289   |

Here the complexity of the program is a bit higher since we need to process 4 inputs and output only one (4:1 mapping). However, the size of the ouput image is 4 times smaller. We notice that the Tesla K40c is a bit slower than the GeForce GTX 680 by 34.41%. We can conclude that the level of complexity of a program may affect the GPU's performance depending of the GPU used.

![Input image](Jaguar.png)  
The input image used for the pooling performance tests.

![Output image](JaguarPooled.png)  
The output image from the pooling performance tests.

### 1.4. Convolution

|NVIDIA GPU     |Time (ms)|
|---------------|---------|
|GeForce GTX 680|0.8109   |
|Tesla K40c     |1.0864   |

Notice here again that the complexity of the program is higher since we map 9 input pixels to 1 out pixel. However, the output image is only slightly smaller. Similar to pooling, convolution on the K40c runs 33.97% slower than the GTX 680. Again, the level of complexity seems to be a factor in the performance of each GPU where, as the level of compelxity increases, the Tesla K40c looses in performance compared to the GeForce GTX 680.

![Input image](JustDoIt.png)  
The input image used for the convolution performance tests.

![Output image](JustDoItConvolved.png)  
The output image from the convolution performance tests.

## 2. Finite element synthesis

### 2.0. CUDA implementation

In the CUDA implementation of lab 2, instead of using a texture object, we used a surface object in `grid.cu`. There are a few advantages of using this type of object. First of all, this makes the read and write operations easier for the GPU. It is also more fit for the purpose of this lab since it is designed for multidimensional arrays, which makes it a better model for our drum simulation.

### 2.1. Performance analysis

We also used two differenet GPU for this lab and analyzed each performance with respect to a 4 by 4 grid and a 512 by 512 grid. 
For the timing analysis, as done in lab 2, we used the `time` bash command to measure our time.

### 2.2. 4 by 4 grid

|NVIDIA GPU     |Time (seconds)|
|---------------|--------------|
|GeForce GTX 680|0.314         |
|Tesla K40c     |0.383         |

Very small simulation, how does that affect the performance difference. 21.97%

For a very small simulation like a 4 by4 grid, the timming a pretty similar. We notice that the Tesla K40x is 21.97% slower than the GeForce GTX 680. However, the size grid makes it probable that the overhead of the K40c is responsible for its slowness compared to the GTX 680.

### 2.2. 512 by 512 grid

|NVIDIA GPU     |Time (seconds)|
|---------------|--------------|
|GeForce GTX 680|1.039         |
|Tesla K40c     |0.964         |

Larger simulation, how does that affect the performance difference.

Here we see that for a larger simulation such as a 512 by 512 grid, the Tesla K40c outperforms the GTX 680 by a slight 7.22%. WE can conclude that both GPU performs approximately the same way on a larger scale.
