#include <stdio.h>
#include "apogpu.h"

__device__ __constant__ int d_delay_length_x_channels;
__device__ __constant__ float d_decay;
__device__ __constant__ int d_samples;

__global__ void gainKernel(float* data_d) {
   float gain = 0.5f;
   unsigned int idx = blockIdx.x*BLOCK_SIZE + threadIdx.x;
   data_d[idx] = data_d[idx] * gain;
   return;
}

__global__ void lowPassKernel(float* data_d) {
   float h = 0.04f;
   unsigned int idx = blockIdx.x*BLOCK_SIZE + threadIdx.x;
   if(idx<25) return;

   float x = 0;
   for(int i=0;i<25;++i) {
      x += data_d[idx-i]*h;
   }
   data_d[idx] = x;
   return;
}

__global__ void delayKernel(float* data_d, int base) {
   unsigned int idx = base + blockIdx.x*BLOCK_SIZE + threadIdx.x;
   if(idx >= d_samples) return;
   data_d[idx+(d_delay_length_x_channels)] += data_d[idx]*d_decay;
   return;
}

void cudasafe( cudaError_t error, char* message) {
   if(error!=cudaSuccess) { fprintf(stderr,"ERROR: %s : %s\n",message,cudaGetErrorString(error)); exit(-1); }
}

void launchGainKernel(float* data_d, int samples) {
   // Stage A:  Setup the kernel execution configuration parameters
   dim3 dimGrid(samples/BLOCK_SIZE,1,1);
   dim3 dimBlock(BLOCK_SIZE,1,1);

   // Stage B: Launch the kernel!! -- using the appropriate function arguments
   gainKernel<<<dimGrid, dimBlock>>>(data_d);


   return;
}

void launchLowPassKernel(float* data_d, int samples) {
   // Stage A:  Setup the kernel execution configuration parameters
   dim3 dimGrid(samples/BLOCK_SIZE,1,1);
   dim3 dimBlock(BLOCK_SIZE,1,1);

   // Stage B: Launch the kernel!! -- using the appropriate function arguments
   lowPassKernel<<<dimGrid, dimBlock>>>(data_d);
   return;
}

void launchDelayKernel(float* data_d, int channels, int samples, float decay, int delay_length) {
   int i;
   int delay_length_x_channels = delay_length*channels;
   cudasafe(cudaMemcpyToSymbol("d_delay_length_x_channels",&delay_length_x_channels, sizeof(int)),"cudaMemcpyToSymbol");
   cudasafe(cudaMemcpyToSymbol("d_decay",&decay,sizeof(float)),"cudaMemcpyToSymbol");
   cudasafe(cudaMemcpyToSymbol("d_samples",&samples,sizeof(int)),"cudaMemcpyToSymbol");
   
   for(i=0;i<samples;i+=delay_length) {
      // Stage A:  Setup the kernel execution configuration parameters
      dim3 dimGrid(delay_length/BLOCK_SIZE,1,1);
      dim3 dimBlock(BLOCK_SIZE,1,1);

      // Stage B: Launch the kernel!! -- using the appropriate function arguments
      delayKernel<<<dimGrid, dimBlock>>>(data_d, i);
   }

   return;
}

void gpusetup(float *data, int channels, int sample_rate, int samples) {
   float *data_d = NULL;
   float time;
   cudaEvent_t start, stop;

   printf("frames: %d\n",samples);

   cudasafe(cudaEventCreate(&start),"cudaEventCreate");
   cudasafe(cudaEventCreate(&stop),"cudaEventCreate");
   cudasafe(cudaEventRecord(start, 0),"cudaEventRecord");

   // Allocate device memory and Transfer host arrays M and N
   cudasafe(cudaMalloc(&data_d, sizeof(float)*samples),"cudaMalloc");

   printf("gpusetup: %f\n",data[0]);

   cudasafe(cudaMemcpy(data_d, data, sizeof(float)*samples, cudaMemcpyHostToDevice),"cudaMempy");

   //launchGainKernel(data_d, samples);
   //launchDelayKernel(data_d, channels, samples, 0.5f, (int)200*(sample_rate/1000));
   launchLowPassKernel(data_d, samples);

   cudasafe(cudaMemcpy(data, data_d, sizeof(float)*samples, cudaMemcpyDeviceToHost),"cudaMemcpy");

   cudasafe(cudaEventRecord(stop, 0),"cudaEventRecord");
   cudasafe(cudaEventSynchronize(stop),"cudaEventSynchronize");
   cudasafe(cudaEventElapsedTime(&time, start, stop),"cudaEvenElapsedTime");

   printf("gpusetup: %f\n",data[0]);

   printf("Time to generate (gpu):  %f ms \n", time);

   int chan,k;
   int delay_length = 200*(sample_rate/1000);
   int globalcount=0;
   float decay  = 0.5;

/*
   cudasafe(cudaEventCreate(&start),"cudaEventCreate");
   cudasafe(cudaEventCreate(&stop),"cudaEventCreate");
   cudasafe(cudaEventRecord(start, 0),"cudaEventRecord");

   for (chan = 0 ; chan < channels ; chan ++) {
      for (k = chan ; k+(delay_length*channels) < samples; k+= channels) {
         data[k+(delay_length*channels)] += data[k]*decay;
         globalcount++;
      }
   }

   cudasafe(cudaEventRecord(stop, 0),"cudaEventRecord");
   cudasafe(cudaEventSynchronize(stop),"cudaEventSynchronize");
   cudasafe(cudaEventElapsedTime(&time, start, stop),"cudaEvenElapsedTime");

   printf("Time to generate (cpu):  %f ms \n", time);
*/
   // End of solution Part 3 ============================================


   // Free device matrices
   cudasafe(cudaFree(data_d),"cudaFree");
}
