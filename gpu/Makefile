NVCC = nvcc -arch=compute_20
INCLUDES = -I/home/uahcls14/local/include
LIBS = -L/home/uahcls14/local/lib -lsndfile

all: apogpu

apogpu.o: apogpu.cu
	$(NVCC) -c $< ${INCLUDES} -o $@

gpu_apogpu.o: gpu_apogpu.cu
	$(NVCC) -c $< ${INCLUDES} -o $@

apogpu: apogpu.o gpu_apogpu.o
	$(NVCC) $^ -o $@ ${LIBS}

clean:
	rm -rf *.o apogpu
