 #include "stdio.h"

// printf() is only supported
// for devices of compute capability 2.0 and higher

__global__ void gainKernel(float* data_d) {
   float gain = 0.5f;
   unsigned int idx = blockIdx.x*32 + threadIdx.x;
   data_d[idx] = data_d[idx] * gain;
   return;
}

void gpusetup(float *data, int channels, int samples) {
   float *data_d = NULL;

   printf("frames: %d\n",samples);

   // Allocate device memory and Transfer host arrays M and N
   cudaMalloc(&data_d, sizeof(float)*samples);

   cudaMemcpy(data_d, data, sizeof(float)*samples, cudaMemcpyHostToDevice);
   //cudaMemcpy(data, data_d, samples, cudaMemcpyHostToDevice);

   // Stage A:  Setup the kernel execution configuration parameters
   dim3 dimGrid(samples/32,1,1);
   dim3 dimBlock(32,1,1);

   printf("gpusetup: %f\n",data[0]);

   // Stage B: Launch the kernel!! -- using the appropriate function arguments
   gainKernel<<<dimGrid, dimBlock>>>(data_d);

   cudaMemcpy(data, data_d, sizeof(float)*samples, cudaMemcpyDeviceToHost);
   //cudaMemcpy(data_d, data, samples, cudaMemcpyDeviceToHost);

   printf("gpusetup: %f\n",data[0]);
   if(cudaGetLastError() != cudaSuccess) { printf("error!\n"); }

   // End of solution Part 3 ============================================


   // Free device matrices
   cudaFree(data_d);
}
