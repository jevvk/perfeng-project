#include <cuda.h>
#include "kernels.h"
#include <stdio.h>

#define uchar unsigned char
#define RGB_STRENGTH 0.5

const int threadBlockWidth  = 16;
const int threadBlockHeight = 16;

static void checkCudaCall(cudaError_t result) {
    if (result != cudaSuccess) {
        printf("Error:: %s: %s\n", cudaGetErrorName(result), cudaGetErrorString(result));
        exit(1);
    }
}

__device__ void cu_copyc(uchar* in, int channels, uchar* out, int v) {
    for (int i = 0; i < channels; i++) {
        out[v * channels + i] = in[v * channels + i];
    }
}

__device__ int cu_argmin(uchar* grad, int a, int b) {
    return grad[a] > grad[b] ? b : a;
}

__device__ int cu_argmax(uchar* grad, int a, int b) {
    return grad[a] < grad[b] ? b : a;
}

__device__ int cu_argmin3(uchar* grad, int a, int b, int c) {
    return cu_argmin(grad, cu_argmin(grad, a, b), c);
}

__device__ int cu_argmax3(uchar* grad, int a, int b, int c) {
    return cu_argmax(grad, cu_argmax(grad, a, b), c);
}

__device__ float cu_clamp(float val, float min_val, float max_val) {
    return max(min(val, max_val), min_val);
}

__device__ uchar cu_blend(uchar base, uchar a, uchar b, uchar c) {
    return (1.0 - RGB_STRENGTH) * base + RGB_STRENGTH * (a + b + c) / 3.0;
}
  
__device__ void cu_blendc(uchar* in, int channels, uchar* out, int base, int a, int b, int c) {
    for (int i = 0; i < channels; i++) {
        out[base * channels + i] = cu_blend(in[base * channels + i], in[a * channels + i], in[b * channels + i], in[c * channels + i]);
    }
}

__global__ void cu_sobel(uchar* in, int width, int height, uchar* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    float kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    float ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
  
    if (i < width - 1 && j < height - 1 && i > 0 && j > 0) {
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
}

__global__ void cu_push_rgb(uchar* data, uchar* grad, uchar* out, uchar* out_grad, int width, int height, int channels) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || i >= height - 1 || j == 0 || j >= width - 1) return;

    int c = i * width + j;

    int tl = (i - 1) * width + (j - 1);
    int tr = (i + 1) * width + (j - 1);

    int bl = (i - 1) * width + (j + 1);
    int br = (i + 1) * width + (j + 1);
    
    int t = i * width + (j - 1);
    int b = i * width + (j + 1);

    int l = (i - 1) * width + j;
    int r = (i + 1) * width + j;

    int min, max;

    // vertical push top -> bottom
    min = cu_argmin3(grad, tl, t, tr);
    max = cu_argmax3(grad, bl, b, br);

    if (grad[min] > grad[max]) {
        cu_blendc(data, channels, out, c, tl, t, tr);
        out_grad[c] = cu_blend(grad[c], grad[tl], grad[t], grad[tr]);
        return;
    }

    // vertical push bottom -> top
    min = cu_argmin3(grad, bl, b, br);
    max = cu_argmax3(grad, tl, t, tr);

    if (grad[min] > grad[max]) {
        cu_blendc(data, channels, out, c, bl, b, br);
        out_grad[c] = cu_blend(grad[c], grad[bl], grad[b], grad[br]);
        return;
    }

    // horizontal push left -> right
    min = cu_argmin3(grad, tl, l, bl);
    max = cu_argmax3(grad, tr, r, br);

    if (grad[min] > grad[max]) {
        cu_blendc(data, channels, out, c, tl, l, bl);
        out_grad[c] = cu_blend(grad[c], grad[tl], grad[l], grad[bl]);
        return;
    }

    // horizontal push right -> left
    min = cu_argmin3(grad, tr, r, br);
    max = cu_argmax3(grad, tl, l, bl);

    if (grad[min] > grad[max]) {
        cu_blendc(data, channels, out, c, tr, r, br);
        out_grad[c] = cu_blend(grad[c], grad[tr], grad[r], grad[br]);
        return;
    }

    // diagonal push top right -> bottom left
    min = cu_argmin3(grad, t, c, r);
    max = cu_argmax3(grad, l, bl, b);

    if (grad[min] > grad[c] && grad[c] > grad[max]) {
        cu_blendc(data, channels, out, c, t, tr, r);
        out_grad[c] = cu_blend(grad[c], grad[t], grad[tr], grad[r]);
        return;
    }

    // diagonal push bottom left -> top right
    min = cu_argmin3(grad, b, c, l);
    max = cu_argmax3(grad, r, tr, t);

    if (grad[min] > grad[c] && grad[c] > grad[max]) {
        cu_blendc(data, channels, out, c, b, bl, l);
        out_grad[c] = cu_blend(grad[c], grad[b], grad[bl], grad[l]);
        return;
    }

    // diagonal push top left -> bottom right
    min = cu_argmin3(grad, t, c, l);
    max = cu_argmax3(grad, r, br, b);

    if (grad[min] > grad[c] && grad[c] > grad[max]) {
        cu_blendc(data, channels, out, c, t, tl, l);
        out_grad[c] = cu_blend(grad[c], grad[t], grad[tl], grad[l]);
        return;
    }

    // diagonal push bottom right -> top left
    min = cu_argmin3(grad, b, c, r);
    max = cu_argmax3(grad, l, tl, t);

    if (grad[min] > grad[c] && grad[c] > grad[max]) {
        cu_blendc(data, channels, out, c, b, br, r);
        out_grad[c] = cu_blend(grad[c], grad[b], grad[br], grad[r]);
        return;
    }

    cu_copyc(data, channels, out, c);
    out_grad[c] = grad[c];
}

__global__ void copy_horizontal(uchar* img, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 0 || i >= width - 1) return;

    img[(height - 1) * width + i] = img[(height - 2) * width + i];
    img[i]                        = img[width + i];
}

__global__ void copy_vertical(uchar* img, int width, int height) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > height - 1) return;

    img[i * width] = img[i * width + 1];
    img[i * width + (width - 2)] = img[i * width + (width - 1)];
}


void create_device_image(void** ptr, int size) {
    checkCudaCall(cudaMalloc(ptr, size));
    if (ptr == NULL) { printf("Error while allocating image of size %d.\n", size); exit(1); }
}   

void copy_to_device(void* device, void* host, int size) {
    checkCudaCall(cudaMemcpy(device, host, size, cudaMemcpyHostToDevice));
}

void copy_from_device(void* host, void* device, int size) {
    checkCudaCall(cudaMemcpy(host, device, size, cudaMemcpyDeviceToHost));
}


void sobel_kernel(uchar* in, int width, int height, uchar* out) {
    // Split input into 16x16 (256 threads per grid)
    dim3 grid(width / threadBlockWidth + 1, height / threadBlockHeight + 1);
    dim3 block(threadBlockWidth, threadBlockHeight);


    cu_sobel<<<grid, block>>>(in, width, height, out);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());


    copy_horizontal<<<width - 2, 1>>>(out, width, height);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());

    copy_vertical<<<height, 1>>>(out, width, height);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());

}

void push_rgb_kernel(uchar* data, uchar* grad, uchar* out, uchar* out_grad, int width, int height, int channels) {

    // Split input into 16x16 (256 threads per grid)
    dim3 grid(width / threadBlockWidth + 1, height / threadBlockHeight + 1);
    dim3 block(threadBlockWidth, threadBlockHeight);

    cu_push_rgb<<<grid, block>>>(data, grad, out, out_grad, width, height, channels);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());

    // copy_horizontal<<<width - 2, 1>>>(out, width, height);
    // cudaDeviceSynchronize();
    // copy_vertical<<<1, height>>>(out, width, height);
    // cudaDeviceSynchronize();

}