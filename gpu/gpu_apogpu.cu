#include <stdio.h>
#include "apogpu.h"

__global__ void gainKernel(float* data_d) {
   float gain = 0.5f;
   unsigned int idx = blockIdx.x*BLOCK_SIZE + threadIdx.x;
   data_d[idx] = data_d[idx] * gain;
   return;
}

__global__ void delayKernel(float* data_d, int channels, int samples, float decay, int delay_length, int base) {
   unsigned int idx = base + blockIdx.x*BLOCK_SIZE + threadIdx.x;
   if(idx >= samples) return;
   data_d[idx+(delay_length*channels)] += data_d[idx]*decay;
   return;
}

void launchGainKernel(float* data_d, int samples) {
   // Stage A:  Setup the kernel execution configuration parameters
   dim3 dimGrid(samples/BLOCK_SIZE,1,1);
   dim3 dimBlock(BLOCK_SIZE,1,1);

   // Stage B: Launch the kernel!! -- using the appropriate function arguments
   gainKernel<<<dimGrid, dimBlock>>>(data_d);


   return;
}

void launchDelayKernel(float* data_d, int channels, int samples, float decay, int delay_length) {
   int i;
   for(i=0;i<samples;i+=delay_length) {
      // Stage A:  Setup the kernel execution configuration parameters
      dim3 dimGrid(delay_length/BLOCK_SIZE,1,1);
      dim3 dimBlock(BLOCK_SIZE,1,1);

      // Stage B: Launch the kernel!! -- using the appropriate function arguments
      delayKernel<<<dimGrid, dimBlock>>>(data_d, channels, samples, decay, delay_length, i);
   }

   return;
}

void gpusetup(float *data, int channels, int sample_rate, int samples) {
   float *data_d = NULL;
   float time;
   cudaEvent_t start, stop;

   printf("frames: %d\n",samples);

   cudaEventCreate(&start);
   cudaEventCreate(&stop);
   cudaEventRecord(start, 0);

   // Allocate device memory and Transfer host arrays M and N
   cudaMalloc(&data_d, sizeof(float)*samples);

   printf("gpusetup: %f\n",data[0]);

   cudaMemcpy(data_d, data, sizeof(float)*samples, cudaMemcpyHostToDevice);

   //launchGainKernel(data_d, samples);
   launchDelayKernel(data_d, channels, samples, 0.5f, (int)200*(sample_rate/1000));

   cudaMemcpy(data, data_d, sizeof(float)*samples, cudaMemcpyDeviceToHost);

   cudaEventRecord(stop, 0);
   cudaEventSynchronize(stop);
   cudaEventElapsedTime(&time, start, stop);

   printf("gpusetup: %f\n",data[0]);

   printf("Time to generate:  %f ms \n", time);

   if(cudaGetLastError() != cudaSuccess) { printf("error!\n"); }

   // End of solution Part 3 ============================================


   // Free device matrices
   cudaFree(data_d);
}
