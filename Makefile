# INCLUDES=-I/usr/include/opencv4
LIBRARIES=-L/usr/local/cuda-10.2/lib64 -lcuda -lcudart

SRC=$(wildcard *.c)
SRC_CUDA=$(wildcard *.cu)

OBJ=$(patsubst %.cpp,%.o,$(SRC))
OBJ_CUDA=$(patsubst %.cu,%.o,$(SRC_CUDA))

CC=gcc
NVCC=nvcc

NVCCFLAGS=-arch=sm_61 -O3

comic-upscaler: main.cpp $(OBJ_CUDA)
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) $(LIBRARIES) $^ -o $@

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm comic-upscaler
