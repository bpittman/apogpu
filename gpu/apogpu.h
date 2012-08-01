#define BLOCK_SIZE 512

// GPU (Device) Matrix/Matrix Multiplication Prototype
// gpu_matrixmul.cpp
void gpusetup(float *data, int channels, int sample_rate, int samples);

// prototype for DeviceSelect Routine (util.cu)
int DeviceSelect(int device_id);

// prototype for DeviceInfo Routine (util.cu)
void DeviceInfo(int device_id);

