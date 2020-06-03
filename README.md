# Reduction in CUDA

*In this hands-on lab and assignment, you will implement a parallel reduction on a GPU and learn some practical techniques to make it run faster. You may if you wish work in teams of two. To simplify our grading of your assignments, eachteamshouldsubmitonecopyoftheassignment. Besuretoindicatewithwhomyouworked by creating a README ﬁle as part of your submission. For this assignment, there may be a lot of competition for time on the cluster. As such, we strongly recommend that you start early.*

### Part 0: Getting the scaffolding code
We’ve written some skeleton code for you to use for this lab. The baseline code for this assignment is available at this URL: https://github.com/EECS-UCI/hw3.git. To get it, use the same git-clone procedure from homework 1 and 2. If it worked, you’ll have a new directory called hw3.

### Part 1: Naive Parallel Reduction
There are 6 CUDA program ﬁles (.cu ﬁles). For Part 1, you will work with naive.cu – this ﬁle contains a naive implementation of parallel reduction. This implementation suffers from several inefﬁciencies, which you will ﬁx. Before you start compiling naive.cu, ﬁrst type the following commands to load the appropriate CUDA and gcc compilers.

```shell
module load cuda/5.0
module load gcc/4.4.3
```

Then, you can compile the code as follows:

```shell
nvcc naive.cu timer.c -o naive
```
which produces naive executable. Try running this on the login node. You should get an error, because the login node does not have a GPU.
```shell
./naive # <--- Should report an error when run on hpc-login
```
Now try instead to submit a job to the GPU node using the provided cuda.sh ﬁle. It should work, and report some statistics on its performance. Start by opening naive.cu and ﬁnd the function kernel0. From its  speciﬁer, we know that this is a GPU kernel. The content of this function is shown below. The arguments on Line 2 point to the beginning of the input and output arrays. The variable n is the number of input elements.


Line 4 declares an array, scratch, of shared memory. Recall that shared memory in CUDA parlancereferstothescratchpadmemorythatissharedbyallthreadsinathreadblock. Theconstant MAX_THREADS is deﬁned to be the number of threads in each block; thus, declaring scratch to be of this size implies 1 word of shared memory per thread. Lines 6 and 7 compute a global ID for the thread. This ID is stored in i. In lines 9-14, each thread loads an element from the global array into the shared memory. In this case, i serves as both a global thread ID and the index of the input element, input[i], assigned to this thread. The call to __syncthreads__() is a barrier for all threads within the same thread block. Here, it is used to ensure that all thread loads have completed prior to any computation on these data. Lines 16-21 perform the reduction within the thread block. The ﬁgure below shows the mapping of threads to shared memory array indices. Lines 23-25 show that only thread 0 in each block writes the sum back to the output array. Notice that the index into the output array is bid. That is, only thread 0 from each block produces a reduced ouput. By indexing the output array this way, we ensure that the result resides consecutively in memory. This is done so that during the next phase of reduction, data will again be accessed consecutively (coalesced memory access). 1. Compile and run this code, which reports the input vector size, the time to execute the kernel, and an effective bandwidth. Record these data. Explain how the effective bandwidth is being calculated.

### Part 2: Strided Access by Consecutive Threads
The performance of the naive code suffers from divergent warps. In successive iterations of the loop, the number of active threads per warp is halved, causing two problems: (a) divergent control ﬂow between active threads and inactive threads, and (b) under-utilization of the threads in each warp, since every warp in the thread block needs to execute even though the number of active threads in each warp is decreasing. Therefore, we would like to change the code so that at each level of the reduction only consecutively numbered threads remain active even as the number of active threads decreases. The following ﬁgure illustrates this technique. 2. Implement this scheme in kernel1 of stride.cu. Measure and record the resulting performance. How much faster than the initial code is this version? Hint: The kernel is very similar to the naive kernel. You should only need to modify Lines 17-19 of the naive kernel.

### Part 3: Sequential Access by Consecutive Threads
Both of the previous kernels suffer from another inefﬁciency, known as bank conﬂicts. Without going into the gory details, bank conﬂicts essentially occur when the threads of a warp make strided accesses to shared memory. Thus, having a warp’s threads access consecutive words in global memory is also a good policy for shared memory. The most common way to prevent bank conﬂicts is to ensure that the threads of a warp access contiguous words in shared memory. For more information on bank conﬂicts, refer to the CUDA Programming Guide. This particular mapping of threads to shared memory indices is shown in the following ﬁgure.


### Part 4: First Add Before Reduce
All of the above implementations can be further improved. Observe that after each thread loads its global array element into the shared memory, half of them immediately become inactive and takenopartintheactualcomputation. Therefore, toimprovethenumberofthreadsdoinguseful work, we can use half the number of threads we did before and have each thread load and sum 2 elements from the global array instead. This mapping of threads to shared memory indices is shown in the following ﬁgure


### Part 5: Unroll the Last Warp
During the GPU reduction lecture, we saw an optimization technique which lets us unroll the last 6 iterations of the inner loop. This is based on exploiting the fact that instructions are SIMD synchronous within a warp. 5. Implement this scheme in kernel4 of unroll.cu and report the effective bandwidth.
### Part 6: Algorithm Cascading
In this part of the lab, instead of having each thread load 2 elements from the global array, have them load multiple elements and then sum them all up before placing the result into the shared memory. This technique is referred to as algorithm cascading. 6. Implement the algorithm cascading scheme in multiple.cu and report the effective bandwidth. Note: In the main function, we have restricted the maximum number of threads to 256 AND the maximum number of thread blocks to 64. This means that there are at most 16384 threads. If the input
6
size is 8388608 elements, then each thread will have to sum up 512 elements from the global array before storing the sum into the shared memory.
### Extra Credit: Matrix transpose
The repo we’ve provided includes another ﬁle, transpose.cu which contains the skeleton code for transposing a matrix on the GPU. Only the code for creating the input matrix on the host has been given. This means that you’ll have to write the code needed for allocating memory for the matrix on the GPU, as well as communication to and fro. More speciﬁcally, edit the functions void gpuTranspose and __global__ void matTrans without changing any of the provided code. Although you have been given freedom to write your own code for this part, you CANNOT change the data structure of the matrix on the host. You may, however, pad the array to ensure that each row starts from an aligned memory address. In addition to submitting your code, brieﬂy describe how your algorithm works and optimizations attempted. Also, present performance results and a brief discussion of the results.
### Submission
When you’ve written up answers to all of the above questions, turn in your write-up and tarball of your code by uploading it to Canvas. Good luck, and have fun!

