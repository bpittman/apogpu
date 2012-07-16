NVCC = nvcc
INCLUDES = -I/home/uahcls14/local/include

all: apogpu

apogpu.o: apogpu.cu util.cu
	$(NVCC) -c $< ${INCLUDES} -o $@

gpu_apogpu.o: gpu_apogpu.cu
	$(NVCC) -c $< ${INCLUDES} -o $@

util.o: util.cu
	$(NVCC) -c $< ${INCLUDES} -o $@

apogpu: apogpu.o gpu_apogpu.o util.o
	$(NVCC) $^ -o $@

clean:
	rm -rf *.o apogpu
