# Report

Aleksi Sapon-Cousineau - 260581670  
Samuel Cauchon - 260587509

## 0. Parallelization

In both labs, the parallelization strategy is to split the problem (the image for lab 1 or the drum for lab 2) into sub problems (blocks) where each subproblem would be solved by a unit (threads and processes in the previous labs). Here, we used CUDA, which uses the GPU to process the data. In order to make both labs parallelized using CUDA, we created a function called `findBestGridAndBlockDims2D` in `common.h`. The goal of this function is to find the best block and grid partitions given the size of a 2D array, so that the GPU is used to its maximum. The CUDA API provides a function `cudaOccupancyMaxPotentialBlockSize` to help perform this estimation for the GPU in use. Every subproblem block is processed  on a thread block on the GPU, such that we have one thread per pixel in lab 1, and one per finite element in lab 2.

## 1. Image transformations

### 1.0. CUDA implementation

The CUDA implementation we designed for this lab also included a texture object in `transform.cu`. A texture object allows us to access a 2D array instead of a 1D array as the input for the GPU. There area few advantages for using this API. First of all, we make a better use of the cache, which in turn makes our program more efficient and faster. The second advantage is that it better models the input data, since it was designed to work with images. We can access the pixels by their x and y coordinates, and all four components are returned as a single 4-tuple with one function call. This simplifies the code a lot.

### 1.1. Performance analysis

For the timing analysis of each type of conversion, we used the CUDA event timming API. This assures that we only get the time that the GPU utilized to process the data. Furthermore, we also tested each conversion using 2 different GPUs to show the performance of both, and draw a conclusion on the performance of a CUDA paralellized program and the GPU used to run the program. The two GPUs used are the NVIDIA Gefore GTX 680 (consumer GPU) and the NVIDIA Tesla K40c (professional GPU).

### 1.2. Rectification

|NVIDIA GPU     |Time (ms)|
|---------------|---------|
|GeForce GTX 680|0.0803   |
|Tesla K40c     |0.0524   |

Here both GPUs completed the task under a tenth of a milisecond. The Tesla K40c took 34.74% less time. Since it is a 1:1 mapping, where each value of the input is process to ouput one value, and the image is rather small, the complexity of the program is very low. Both GPUs ran the program extremely quickly and are within margin of error.

![Input image](Rooster.png)  
The input image used for the rectification performance tests.

![Output image](RoosterRectified.png)  
The output image from the rectification performance tests.

### 1.3. Pooling

|NVIDIA GPU     |Time (ms)|
|---------------|---------|
|GeForce GTX 680|0.0215   |
|Tesla K40c     |0.0289   |

Here the complexity of the program is a bit higher since we need to process four input pixels to output only one pixel (4:1 mapping). However, the size of the ouput image is four times smaller, so the complexity is actually closer to 1:1. The input image is also still rather small. Again both GPUs completed the task under a tenth of a milisecond. The Tesla K40c took 34.41% more time. Both GPUs ran the program extremely quickly and are within margin of error.

![Input image](Jaguar.png)  
The input image used for the pooling performance tests.

![Output image](JaguarPooled.png)  
The output image from the pooling performance tests.

### 1.4. Convolution

|NVIDIA GPU     |Time (ms)|
|---------------|---------|
|GeForce GTX 680|0.8109   |
|Tesla K40c     |1.0864   |

Notice here again that the complexity of the program is higher since we map nine input pixels to one out pixel. The output image is smaller, but the difference is quite negligeable. This computation ran the longest, taking around one milisecond. The Tesla K40c took 33.97% more time. Although the absolute time difference is larger, it's hard to say if the Tesla K40c was slower or if other components of the program made a difference. It's possible that we are within margin of error, or that the CPU and bus transfers speeds made a difference.

![Input image](JustDoIt.png)  
The input image used for the convolution performance tests.

![Output image](JustDoItConvolved.png)  
The output image from the convolution performance tests.

## 2. Finite element synthesis

### 2.0. CUDA implementation

In the CUDA implementation of lab 2, instead of using a texture object, we used a surface object in `grid.cu`. There are a few advantages of using this type of object. First of all, this makes the read and write operations easier for the GPU. It is also more fit for the purpose of this lab since it is designed for multidimensional arrays, which makes it a better model for our drum simulation. It also improves caching performance.

The CUDA implementation is very different to the MPI one. Here, we used no synchronization or atomics at all. Each thread remains completely independent from the others. We found a way to compute a whole iteration as a single kernel call, by collapsing the update equations for the edge and corner nodes. This has the trade-off that the new values of the nodes right next to the edges are calculated twice, although they still only represent a small fraction of the total nodes. If using the original equations, the cost of updating the edges and corners is less than that of the middle nodes. But if we collapse them we have the same "middle" cost for all nodes, so it does not make a big difference. Also synchronization and atomics have a cost associated to them, especially accross blocks, potentially higher than the few redundant updates.

### 2.1. Performance analysis

We also used two differenet GPUs for this lab and analyzed each performance with respect to a 4 by 4 grid and a 512 by 512 grid. For the timing analysis, as done in lab 2, we used the `time` bash command's real time output.

### 2.2. 4 by 4 grid

|NVIDIA GPU     |Time (seconds)|
|---------------|--------------|
|GeForce GTX 680|0.314         |
|Tesla K40c     |0.383         |

For a very small simulation like a 4 by 4 grid, the timming a pretty similar. We notice that the Tesla K40x is 21.97% slower than the GeForce GTX 680. However, the tiny grid size makes it probable that the overhead of the K40c is responsible for its slowness compared to the GTX 680. This, we are within margin of error.

### 2.2. 512 by 512 grid

|NVIDIA GPU     |Time (seconds)|
|---------------|--------------|
|GeForce GTX 680|1.039         |
|Tesla K40c     |0.964         |

Here we see that for a larger simulation such as a 512 by 512 grid, the Tesla K40c outperforms the GTX 680 by a slight 7.22%. We can conclude that both GPU performs approximately the same way on a larger scale.

## 3. Conclusion

From the results that we gathered in both labs using CUDA, we can conclude that, overall, both GPUs have the same performance in all tests. The maximum difference in processing time for both GPUs is 34.74%. However, the difference is in the fraction of a millisecond, which is not significiant. This might have happened because the programs that we developped are not complex enough and/or do not process enough data. That is to say, we are never using the GPUs to their 100%. The extra threads in the Tesla K40c are simply never used, and all work to be scheduled is done immediately. Then the difference in performance would come from the performance of a single thread, since the runtime will be the maximum runtime over all threads. Also we subdivided the work as much as we could, with a pixel per thread for lab 1 and a finite element per thread in lab 2. So we cannot achieve higher utilization.
