# INCLUDES=-I/usr/include/opencv4
LIBRARIES=-L/usr/local/cuda-10.2/lib64 -lcuda -lcudart

CC=gcc
NVCC=nvcc

NVCCFLAGS=-O3

bench: bench.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIBRARIES) $^ -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm bench
