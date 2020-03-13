#include <stdlib.h>
#include <stdio.h>

#include "cuda_utils.h"
#include "timer.c"

typedef float dtype;

const int T_SIZE = 32;
const int STRIDE = 8;


__global__
void matTrans(dtype* AT, dtype* A, int N)  {
	/* Fill your code here */
  /*
   This is where the actual transpose happens, AT seems like matrix that is the output
   A is the matrix that needs to be transposed, and N is the length.
   The __global__ tells the compiler that this will run on the GPU
  */
  __shared__  dtype scratch[T_SIZE][T_SIZE]; //N should be the size of the tile

  //each thread block should work on a separate tile
  // having a smaller thread count than the tile size is better for some reason
  // Location of x and y is the location on the 2D array
  // Width is the total size of the array
  unsigned int x = blockIdx.x * T_SIZE + threadIdx.x;
  unsigned int y = blockIdx.y * T_SIZE + threadIdx.y;
  unsigned int W = gridDim.x * T_SIZE;

  //stride is the stride size for the rows in the matrix
  // basically this part just copies to local shared mem
  for (int i = 0 ; i < T_SIZE ; i+= STRIDE){
    // from idata is 1 dimensional mapping from a 2D array
      // assert(threadIdx.y + i < T_SIZE);
      // assert(threadIdx.x < T_SIZE);
      // assert((y+i)* width + x < N);
      scratch [threadIdx.y + i][threadIdx.x] = A[(y+i)* W + x];
  }
  __syncthreads();

  x = blockIdx.y * T_SIZE + threadIdx.x;
  y = blockIdx.x * T_SIZE + threadIdx.y;

  for(int i = 0 ; i < T_SIZE; i+= STRIDE){
    // assert(threadIdx.y + i < T_SIZE);
    // assert(threadIdx.x < T_SIZE);
    // assert((y+i)* width + x < N);
    AT[(y + i) * W + x] = scratch[threadIdx.x][threadIdx.y + i];
  }
}

void
parseArg (int argc, char** argv, int* N)
{
	if(argc == 2) {
		*N = atoi (argv[1]);
		assert (*N > 0);
	} else {
		fprintf (stderr, "usage: %s <N>\n", argv[0]);
		exit (EXIT_FAILURE);
	}
}


void
initArr (dtype* in, int N)
{
	int i;

	for(i = 0; i < N; i++) {
		in[i] = (dtype) rand () / RAND_MAX;
	}
}

void
cpuTranspose (dtype* A, dtype* AT, int N)
{
	int i, j;

	for(i = 0; i < N; i++) {
		for(j = 0; j < N; j++) {
			AT[j * N + i] = A[i * N + j];
		}
	}
}

int
cmpArr (dtype* a, dtype* b, int N)
{
	int cnt, i;

	cnt = 0;
	for(i = 0; i < N; i++) {
		if(abs(a[i] - b[i]) > 1e-6) cnt++;
	}

	return cnt;
}



void
gpuTranspose (dtype* A, dtype* AT, int N)
{
  unsigned int s = N * N * sizeof(dtype);
  /*
  This is where the memory needs to be allocated.
  I want to allocate the device memory
  */
  dtype *d_iA;
  dtype *d_oA;
  dim3 gb(N/T_SIZE, N/T_SIZE, 1);
  dim3 tb(T_SIZE, STRIDE, 1);

  CUDA_CHECK_ERROR (cudaMalloc (&d_iA, s));
	CUDA_CHECK_ERROR (cudaMalloc (&d_oA, s));
  CUDA_CHECK_ERROR (cudaMemcpy (d_iA, A, s,cudaMemcpyHostToDevice));

  struct stopwatch_t* timer = NULL;
  long double t_gpu;


  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();
	/* run your kernel here */
  // triple angle brackets mark a call from host to device, launching the kernel
  // launch kernel with this number of blocks,launch kernel with this number of threads
  //warm up
  matTrans <<<gb,tb>>>(d_oA, d_iA,N);
  cudaThreadSynchronize ();
  matTrans <<<gb,tb>>>(d_oA, d_iA,N);
  cudaThreadSynchronize ();
  stopwatch_start (timer);
    matTrans<<<gb,tb>>>(d_oA, d_iA,N);
  cudaThreadSynchronize ();
  t_gpu = stopwatch_stop (timer);
  CUDA_CHECK_ERROR (cudaMemcpy (AT, d_oA, s,
				cudaMemcpyDeviceToHost));
  fprintf (stdout, "Time to execute sequential index GPU reduction kernel: %Lg secs\n", t_gpu);
  fprintf (stderr, "GPU transpose: %Lg secs ==> %Lg billion elements/second\n",
           t_gpu, (N * N) / t_gpu * 1e-9 );
 double bw = (N * N * sizeof(dtype)) / (t_gpu * 1e9);
 fprintf (stdout, "Effective bandwidth: %.2lf GB/s\n", bw);

}

int
main(int argc, char** argv)
{
  /* variables */
	dtype *A, *ATgpu, *ATcpu;
  int err;

	int N;

  struct stopwatch_t* timer = NULL;
  long double t_cpu;


	N = -1;
	parseArg (argc, argv, &N);

  /* input and output matrices on host */
  /* output */
  ATcpu = (dtype*) malloc (N * N * sizeof (dtype));
  ATgpu = (dtype*) malloc (N * N * sizeof (dtype));

  /* input */
  A = (dtype*) malloc (N * N * sizeof (dtype));

	initArr (A, N * N);

	/* GPU transpose kernel */
  printf("GPU Transpose\n");
	gpuTranspose (A, ATgpu, N);

  /* Setup timers */
  stopwatch_init ();
  timer = stopwatch_create ();

	stopwatch_start (timer);
  /* compute reference array */
	cpuTranspose (A, ATcpu, N);
  t_cpu = stopwatch_stop (timer);
  fprintf (stderr, "Time to execute CPU transpose kernel: %Lg secs\n",
           t_cpu);


  /* check correctness */
	err = cmpArr (ATgpu, ATcpu, N * N);
	if(err) {
		fprintf (stderr, "Transpose failed: %d\n", err);
	} else {
		fprintf (stderr, "Transpose successful\n");
	}

	free (A);
	free (ATgpu);
	free (ATcpu);

  return 0;
}
