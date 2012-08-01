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

__global__ void lowPassKernel(float* data_d, float* results_d, int channels) {
   extern __shared__ float data_s[];
   float h = 0.03125f;
   unsigned int tidx = threadIdx.x;
   unsigned int idx = blockIdx.x*BLOCK_SIZE + tidx;

   //load the last 256 frames into shared memory
   for(int i=0;i<channels;++i) {
      data_s[(tidx+32)*channels+i] = data_d[(idx*channels)+i];
   }

   if(idx<32*channels) {
      __syncthreads();
      return;
   }

   //load the first 32 frames into shared memory
   if(tidx<32) {
      for(int i=0;i<channels;++i) {
         data_s[(tidx*channels)+i] = data_d[((idx-32)*channels)+i];
      }
   }

   __syncthreads();

   float x = 0;
   for(int i=0;i<32*channels;i+=channels) {
      x += data_s[tidx-i+(32*channels)]*h;
   }
   results_d[idx] = x;
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

   cudaThreadSynchronize();
   cudaError_t error = cudaGetLastError();
   if(error != cudaSuccess) {
      printf("gainKernel error: %s\n", cudaGetErrorString(error));
   }

   return;
}

void launchLowPassKernel(float* data_d, float* results_d, int samples, int channels) {
   // Stage A:  Setup the kernel execution configuration parameters
   dim3 dimGrid(samples/BLOCK_SIZE,1,1);
   dim3 dimBlock(BLOCK_SIZE,1,1);
   int shared_size = ((BLOCK_SIZE+32)*channels)*sizeof(float);

   // Stage B: Launch the kernel!! -- using the appropriate function arguments
   lowPassKernel<<<dimGrid, dimBlock, shared_size>>>(data_d, results_d, channels);

   cudaThreadSynchronize();
   cudaError_t error = cudaGetLastError();
   if(error != cudaSuccess) {
      printf("lowPassKernel error: %s\n", cudaGetErrorString(error));
   }

   return;
}

void launchDelayKernel(float* data_d, int channels, int samples, float decay, int delay_length) {
   int i;
   int delay_length_x_channels = delay_length*channels;
   cudasafe(cudaMemcpyToSymbol("d_delay_length_x_channels",&delay_length_x_channels, sizeof(int)),"cudaMemcpyToSymbol");
   cudasafe(cudaMemcpyToSymbol("d_decay",&decay,sizeof(float)),"cudaMemcpyToSymbol");
   cudasafe(cudaMemcpyToSymbol("d_samples",&samples,sizeof(int)),"cudaMemcpyToSymbol");
   
   for(i=0;i<samples;i+=delay_length_x_channels) {
      // Stage A:  Setup the kernel execution configuration parameters
      dim3 dimGrid(delay_length_x_channels/BLOCK_SIZE,1,1);
      dim3 dimBlock(BLOCK_SIZE,1,1);

      // Stage B: Launch the kernel!! -- using the appropriate function arguments
      delayKernel<<<dimGrid, dimBlock>>>(data_d, i);

      cudaThreadSynchronize();
      cudaError_t error = cudaGetLastError();
      if(error != cudaSuccess) {
         printf("delayKernel error: %s\n", cudaGetErrorString(error));
      }
   }

   return;
}

void gpusetup(float *data, int channels, int sample_rate, int samples) {
   float *data_d = NULL, *results_d = NULL;
   float time;
   cudaEvent_t start, stop;

   printf("frames: %d\n",samples);

   cudasafe(cudaEventCreate(&start),"cudaEventCreate");
   cudasafe(cudaEventCreate(&stop),"cudaEventCreate");
   cudasafe(cudaEventRecord(start, 0),"cudaEventRecord");

   cudasafe(cudaMalloc(&data_d, sizeof(float)*samples),"cudaMalloc");
   cudasafe(cudaMalloc(&results_d, sizeof(float)*samples),"cudaMalloc");

   printf("gpusetup: %f\n",data[0]);

   cudasafe(cudaMemcpy(data_d, data, sizeof(float)*samples, cudaMemcpyHostToDevice),"cudaMempy");

   launchGainKernel(data_d, samples);
   launchDelayKernel(data_d, channels, samples, 0.5f, (int)256*(sample_rate/1000));
   launchLowPassKernel(data_d, results_d, samples, channels);

   //cudasafe(cudaMemcpy(data, data_d, sizeof(float)*samples, cudaMemcpyDeviceToHost),"cudaMemcpy");
   cudasafe(cudaMemcpy(data, results_d, sizeof(float)*samples, cudaMemcpyDeviceToHost),"cudaMemcpy");

   cudasafe(cudaEventRecord(stop, 0),"cudaEventRecord");
   cudasafe(cudaEventSynchronize(stop),"cudaEventSynchronize");
   cudasafe(cudaEventElapsedTime(&time, start, stop),"cudaEvenElapsedTime");

   printf("gpusetup: %f\n",data[0]);

   printf("Time to generate (gpu):  %f ms \n", time);

   int chan,k,i;
   int delay_length = 256*(sample_rate/1000);
   int globalcount=0;
   float decay  = 0.5;
   float h = 0.03125f;
   float *results=NULL;
   float channel_gain = 0.5f;

   results = (float*)malloc(sizeof(float)*samples);
   if(results == NULL)
   {
      printf("malloc failed!\n");
      return;
   }

   cudasafe(cudaEventCreate(&start),"cudaEventCreate");
   cudasafe(cudaEventCreate(&stop),"cudaEventCreate");
   cudasafe(cudaEventRecord(start, 0),"cudaEventRecord");

   for (chan = 0 ; chan < channels ; chan ++) {
      for (k = chan ; k+(delay_length*channels) < samples; k+= channels) {
         data[k] *= channel_gain;
      }
   }

   for (chan = 0 ; chan < channels ; chan ++) {
      for (k = chan ; k+(delay_length*channels) < samples; k+= channels) {
         data[k+(delay_length*channels)] += data[k]*decay;
         globalcount++;
      }
   }

    for(k=0;k<samples;++k) {
        if(k<32*channels) continue;
        float x = 0;
        for(i=0; i<32*channels; i+=channels) {
            x += data[(k-i)]*h;
        }
        results[k] = x;
    }


   cudasafe(cudaEventRecord(stop, 0),"cudaEventRecord");
   cudasafe(cudaEventSynchronize(stop),"cudaEventSynchronize");
   cudasafe(cudaEventElapsedTime(&time, start, stop),"cudaEvenElapsedTime");

   printf("Time to generate (cpu):  %f ms \n", time);

   // End of solution Part 3 ============================================


   // Free device matrices
   cudasafe(cudaFree(data_d),"cudaFree");
}
