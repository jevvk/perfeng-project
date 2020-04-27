#include <cuda.h>
#include "kernels.h"
#include <stdio.h>

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("Error:: %s: %s\n", cudaGetErrorName(result), cudaGetErrorString(result));
        exit(1);
    }
}

__device__ float cu_clamp(float val, float min_val, float max_val) {
    return fmax(fmin(val, max_val), min_val);
}

__global__ void cu_sobel(unsigned char* in, int width, int height, unsigned char* out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    float kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
  
    if (i < height - 1 && j < width - 1 && i > 0 && j > 0) {
        float mag_x = 0;
        float mag_y = 0;

        for (int ki = 0; ki < 3; ki++) {
            for (int kj = 0; kj < 3; kj++) {
                mag_x += kx[ki][kj] * in[(i + ki - 1) * width + j + kj - 1];
                mag_y += ky[ki][kj] * in[(i + ki - 1) * width + j + kj - 1];
            }
        }
    
        out[i * width + j] = 255.0 - cu_clamp(sqrt(mag_x * mag_x + mag_y * mag_y), 0, 255.0);
      
    }
    // expand the image by 1px
    // not sure, we could change this
  
    // for (int i = 1; i < width - 1; i ++) {
    //     out[i] = out[width + i];
    //     out[(height - 1) * width + i] = out[(height - 2) * width + i];
    // }
  
    // for (int i = 1; i < height - 1; i ++) {
    //     out[i * width] = out[i * width + 1];
    //     out[(i + 1) * width - 1] = out[(i + 1) * width - 2];
    // }
  
    // out[0] = (out[1] + out[width]) / 2;
    // out[width - 1] = (out[width - 2] + out[2 * width - 1]) / 2;
  
    // out[(height - 1) * width] = (out[(height - 2) * width] + out[(height - 1) * width + 1]) / 2;
    // out[height * width - 1] = (out[height * width - 2] + out[(height - 1) * width - 1]) / 2;
}


void create_device_image(void** ptr, int size) {
    printf("Allocating size %d\n", size);
    checkCudaCall(cudaMalloc(ptr, size));
    if (ptr == NULL) { printf("Error while allocating image of size %d.\n", size); exit(1); }
}   

void copy_to_device(void* device, void* host, int size) {
    printf("Copying %d to device\n", size);
    checkCudaCall(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
}

void copy_from_device(void* host, void* device, int size) {
    printf("Copying %d from dvice\n", size);
    checkCudaCall(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
}


void sobel_kernel(unsigned char* in, int width, int height, unsigned char* out) {
    printf("Entered sobel kernel\n");

    int threadBlockHeight = 16;
    int threadBlockWidth  = 16;

    // Split input into 16x16 (256 threads per grid)
    dim3 grid(width / threadBlockWidth + 1, height / threadBlockHeight + 1);
    dim3 block(threadBlockWidth, threadBlockHeight);

    cu_sobel<<<grid, block>>>(in, width, height, out);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
    printf("Done with sobel kernel\n");
}