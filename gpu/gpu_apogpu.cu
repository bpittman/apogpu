__global__ void gainKernel(int* data) {
   return;
}

void gpusetup(int *data, int channels, int samples) {
   int *data_d;

   // Allocate device memory and Transfer host arrays M and N
   cudaMalloc((void**) &data_d, channels*samples);

   cudaMemcpy(data_d, data, channels*samples, cudaMemcpyHostToDevice);

   // Stage A:  Setup the kernel execution configuration parameters
   dim3 dimGrid(1,1,1);
   dim3 dimBlock(1,1,1);
   // Stage B: Launch the kernel!! -- using the appropriate function arguments
   gainKernel<<<dimGrid, dimBlock>>>(data_d);

   cudaMemcpy(data_d, data, channels*samples, cudaMemcpyDeviceToHost);

   // End of solution Part 3 ============================================


   // Free device matrices
   cudaFree(data_d);
}
