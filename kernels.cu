
#include <cuda.h>
#include "kernels.h"
#include <stdio.h>

#define uchar unsigned char
#define RGB_STRENGTH 0.5
#define BITMASK_RADIUS 5

const int threadBlockWidth  = 16;
const int threadBlockHeight = 16;

uchar* bitmask_array = NULL;

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

__device__ uchar cu_min3(uchar a, uchar b, uchar c) {
    return min(min(a, b), c);
}
  
__device__ uchar cu_max3(uchar a, uchar b, uchar c) {
    return max(max(a, b), c);
}

__device__ float cu_clamp(float val, float min_val, float max_val) {
    return max(min(val, max_val), min_val);
}

__device__ uchar cu_blend(uchar base, uchar a, uchar b, uchar c) {
    return (1.0 - RGB_STRENGTH) * base + RGB_STRENGTH * (a + b + c) / 3.0;
}
  
__device__ uchar cu_blend_lightest(uchar lightest, uchar base, uchar a, uchar b, uchar c) {
    return max(lightest, cu_blend(base, a, b, c));
  }
  

__device__ void cu_blendc(uchar* in, int channels, uchar* out, int base, int a, int b, int c) {
    for (int i = 0; i < channels; i++) {
        out[base * channels + i] = cu_blend(in[base * channels + i], in[a * channels + i], in[b * channels + i], in[c * channels + i]);
    }
}

__global__ void cu_luminance(uchar* in, int width, int height, int channels, uchar* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < 0 || i >= height || j < 0 || j >= width) return;

    out[i * width + j] = 0.3  * in[i * width * channels + j * channels]
                       + 0.58 * in[i * width * channels + j * channels + 1]
                       + 0.11 * in[i * width * channels + j * channels + 2];
}

__global__ void cu_resize(uchar* in, int width, int height, int channels, float scale, uchar* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    
    int new_width = floor(scale * width);
    int new_height = floor(scale * height);
    
    if (i < 0 || i >= new_height || j < 0 || j >= new_width) return;
  
    for (int c = 0; c < channels; c++) {
        float sample_row = (float) i * height / new_height;
        float sample_col = (float) j * width / new_width;

        int sample_tl_row = floor((double) sample_row);
        int sample_tl_col = floor(sample_col);

        int sample_bl_row = sample_tl_row + 1;
        int sample_bl_col = sample_tl_col;

        int sample_tr_row = sample_tl_row;
        int sample_tr_col = sample_tl_col + 1;

        int sample_br_row = sample_tl_row + 1;
        int sample_br_col = sample_tl_col + 1;

        float weight_row = 1.0f - sample_row + sample_tl_row;
        float weight_col = 1.0f - sample_col + sample_tl_col;

        float sample1 = weight_row * in[sample_tl_row * width * channels + sample_tl_col * channels + c];
        if (sample_bl_row < height) {
            sample1 += (1.0f - weight_row) * in[sample_bl_row * width * channels + sample_bl_col * channels + c];
        }

        if (sample_tr_col < width) {
            float sample2 = weight_row * in[sample_tr_row * width * channels + sample_tr_col * channels + c];

            if (sample_br_row < height) {
                sample2 += (1.0f - weight_row) * in[sample_br_row * width * channels + sample_br_col * channels + c];
            }

            sample1 = sample1 * weight_col + (1.0f - weight_col) * sample2;
        }

        out[i * new_width * channels + j * channels + c] = sample1;
    }
}


/*
 * Gaussian3 kernel does smoothing on all three colour channels.
 */
__global__ void cu_gaussian3(uchar* in, int width, int height, uchar* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || i >= height - 1 || j == 0 || j >= width - 1) return;
    
    const float g[3][3] = {{1/16.0, 1/8.0, 1/16.0}, {1/8.0, 1/4.0, 1/8.0}, {1/16.0, 1/8.0, 1/16.0}};

    #pragma unroll
    for (int c = 0; c < 3; c++) {
        float res = 0;

        for (int ki = 0; ki < 3; ki++) {
            for (int kj = 0; kj < 3; kj++) {
                res += g[ki][kj] * in[(i + ki - 1) * width * 3 + (j + kj - 1) * 3 + c];
            }
        }

        out[i * width * 3 + j * 3 + c] = res;
    }
}

/*
 * Gaussian kernel smoothes on a grayscale image with range 0-255.
 * It then computes the difference between the input image and the smoothed pixel.
 * This gives us an edge image normalized to range 0-255
 */
__global__ void cu_gaussian(uchar* in, int width, int height, uchar* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || i >= height - 1 || j == 0 || j >= width - 1) return;    
    
    const float g[3][3] = {{1/16.0, 1/8.0, 1/16.0}, {1/8.0, 1/4.0, 1/8.0}, {1/16.0, 1/8.0, 1/16.0}};
    float res = 0;

    for (int ki = 0; ki < 3; ki++) {
        for (int kj = 0; kj < 3; kj++) {
            res += g[ki][kj] * in[(i + ki - 1) * width + (j + kj - 1)];
        }
    }

    out[i * width + j] = res;
}

__global__ void cu_diff_edge_thresh(uchar* in, uchar* in_smooth, int width, int height, int threshold, uchar* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= height || j >= width) return;    

    uchar res = abs(in[i * width + j] - in_smooth[i * width + j]);
    out[i * width + j] = res > threshold ? 1 : 0;
}


/* 
 * The bitmask kernel looks in a grayscale edge image (0-255) for any values in an area of radius larger than threshold.
 * If any value is large enough, the bitmask sets the value to 1, 0 otherwise.
 * This should allow the more specific use of exclusively pixels around edges and reduce operations.
 */
__global__ void cu_dilate(uchar* in, int width, int height, uchar* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < BITMASK_RADIUS || i >= height - BITMASK_RADIUS || j < BITMASK_RADIUS || j >= width - BITMASK_RADIUS) return;

    uchar kernel[3][3] = { {0, 1, 0}, {1, 1, 1}, {0, 1, 0} };

    #pragma unroll
    for (int ii = 0; ii < 3; ii++) {
        #pragma unroll
        for (int jj = 0; jj < 3; jj++) {
            // hope compiler optimizez this
            if (kernel[ii][jj] == 0) {
                continue;
            }

            if (in[(i + ii - 1) * width + j + jj - 1] == 1) {
                out[i * width + j] = 1;
                return;
            }
        }
    }

    out[i * width + j] = 0;


    // for (int y = -BITMASK_RADIUS; y < BITMASK_RADIUS + 1; y++) {
    //     int ii = i + y;
    //     for (int x = -BITMASK_RADIUS; x < BITMASK_RADIUS + 1; x++) {
    //         int jj = j + x;
    //         if (in[ii * width + jj] == 1) {
    //             out[ii * width + jj]) = 1;
    //         }
    //     }
    // }
    // out[i * width + j] = 0;
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

__global__ void cu_push_rgb(uchar* data, uchar* grad, uchar* out, uchar* out_grad, int width, int height, int channels, uchar* bitmask) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;    
    
    if (i == 0 || i >= height - 1 || j == 0 || j >= width - 1) return;
    
    int c = i * width + j;
    out_grad[c] = grad[c];

    if (bitmask[c] == 0)  {
        cu_copyc(data, channels, out, c);
        return;
    }

    int tl = (i - 1) * width + (j - 1);
    int tr = (i + 1) * width + (j - 1);

    int bl = (i - 1) * width + (j + 1);
    int br = (i + 1) * width + (j + 1);
    
    int t = i * width + (j - 1);
    int b = i * width + (j + 1);

    int l = (i - 1) * width + j;
    int r = (i + 1) * width + j;

    uchar gc = grad[c];

    uchar gtl = grad[tl];
    uchar gtr = grad[tr];

    uchar gbl = grad[bl];
    uchar gbr = grad[br];

    uchar gt = grad[t];
    uchar gb = grad[b];

    uchar gl = grad[l];
    uchar gr = grad[r];

    uchar min, max;

    // vertical push top -> bottom
    min = cu_min3(gtl, gt, gtr);
    max = cu_max3(gbl, gb, gbr);

    if (min > max) {
        cu_blendc(data, channels, out, c, tl, t, tr);
        out_grad[c] = cu_blend(gc, gtl, gt, gtr);
        return;
    }

    // vertical push bottom -> top
    min = cu_min3(gbl, gb, gbr);
    max = cu_max3(gtl, gt, gtr);

    if (min > max) {
        cu_blendc(data, channels, out, c, bl, b, br);
        out_grad[c] = cu_blend(gc, gbl, gb, gbr);
        return;
    }

    // horizontal push left -> right
    min = cu_min3(gtl, gl, gbl);
    max = cu_max3(gtr, gr, gbr);

    if (min > max) {
        cu_blendc(data, channels, out, c, tl, l, bl);
        out_grad[c] = cu_blend(gc, gtl, gl, gbl);
        return;
    }

    // horizontal push right -> left
    min = cu_min3(gtr, gr, gbr);
    max = cu_max3(gtl, gl, gbl);

    if (min > max) {
        cu_blendc(data, channels, out, c, tr, r, br);
        out_grad[c] = cu_blend(gc, gtr, gr, gbr);
        return;
    }

    // diagonal push top right -> bottom left
    min = cu_min3(gt, gc, gr);
    max = cu_max3(gl, gbl, gb);

    if (min > gc && gc > max) {
        cu_blendc(data, channels, out, c, t, tr, r);
        out_grad[c] = cu_blend(gc, gt, gtr, gr);
        return;
    }

    // diagonal push bottom left -> top right
    min = cu_min3(gb, gc, gl);
    max = cu_max3(gr, gtr, gt);

    if (min > gc && gc > max) {
        cu_blendc(data, channels, out, c, b, bl, l);
        out_grad[c] = cu_blend(gc, gb, gbl, gl);
        return;
    }

    // diagonal push top left -> bottom right
    min = cu_min3(gt, gc, gl);
    max = cu_max3(gr, gbr, gb);

    if (min > gc && gc > max) {
        cu_blendc(data, channels, out, c, t, tl, l);
        out_grad[c] = cu_blend(gc, gt, gtl, gl);
        return;
    }

    // diagonal push bottom right -> top left
    min = cu_min3(gb, gc, gr);
    max = cu_max3(gl, gtl, gt);

    if (min > gc && gc > max) {
        cu_blendc(data, channels, out, c, b, br, r);
        out_grad[c] = cu_blend(gc, gb, gbr, gr);
        return;
    }

    cu_copyc(data, channels, out, c);
    out_grad[c] = grad[c];
}

__global__ void cu_push_grad(uchar* in, int width, int height, uchar* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || i >= height - 1 || j == 0 || j >= width - 1) return;

    int c = in[i * width + j];

    int tl = in[(i - 1) * width + (j - 1)];
    int tr = in[(i + 1) * width + (j - 1)];

    int bl = in[(i - 1) * width + (j + 1)];
    int br = in[(i + 1) * width + (j + 1)];
    
    int t = in[i * width + (j - 1)];
    int b = in[i * width + (j + 1)];

    int l = in[(i - 1) * width + j];
    int r = in[(i + 1) * width + j];

    int min, max;

    // vertical push top -> bottom
    min = cu_min3(tl, t, tr);
    max = cu_max3(bl, b, br);

    uchar res = c;
    uchar min_res = c;

    if (min > max) {
        res = cu_blend_lightest(res, c, tl, t, tr);
        min_res = res > min_res ? min_res : res;
    }

    // vertical push bottom -> top
    min = cu_min3(bl, b, br);
    max = cu_max3(tl, t, tr);

    if (min > max) {
        res = cu_blend_lightest(res, c, bl, b, br);
        min_res = res > min_res ? min_res : res;
    }

    // horizontal push left -> right
    min = cu_min3(tl, l, bl);
    max = cu_max3(tr, r, br);

    if (min > max) {
        res = cu_blend_lightest(res, c, tl, l, bl);
        min_res = res > min_res ? min_res : res;
    }

    // horizontal push right -> left
    min = cu_min3(tr, r, br);
    max = cu_max3(tl, l, bl);

    if (min > max) {
        res = cu_blend_lightest(res, c, tr, r, br);
        min_res = res > min_res ? min_res : res;
    }

    // diagonal push top right -> bottom left
    min = cu_min3(t, c, r);
    max = cu_max3(l, bl, b);

    if (min > res && res > max) {
        res = cu_blend_lightest(res, c, t, tr, r);
        min_res = res > min_res ? min_res : res;
    }

    // diagonal push bottom left -> top right
    min = cu_min3(b, c, l);
    max = cu_max3(r, tr, t);

    if (min > res && res > max) {
        res = cu_blend_lightest(res, c, b, bl, l);
        min_res = res > min_res ? min_res : res;
    }

    // diagonal push top left -> bottom right
    min = cu_min3(t, c, l);
    max = cu_max3(r, br, b);

    if (min > res && res > max) {
        res = cu_blend_lightest(res, c, t, tl, l);
        min_res = res > min_res ? min_res : res;
    }

    // diagonal push bottom right -> top left
    min = cu_min3(b, c, r);
    max = cu_max3(l, tl, t);

    if (min > res && res > max) {
        res = cu_blend_lightest(res, c, b, br, r);
        min_res = res > min_res ? min_res : res;
    }

    out[i * width + j] = min_res;
}

__global__ void copy_horizontal(uchar* img, int width, int height, int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i == 0 || i >= width - 1) return;

    #pragma unroll
    for (int n = 0; n < channels; n++) {
        img[((height - 1) * width + i) * channels + n] = img[((height - 2) * width + i) * channels + n];
        img[i * channels + n] = img[(width + i) * channels + n];
    }
}

__global__ void copy_vertical(uchar* img, int width, int height, int channels) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i > height - 1) return;

    #pragma unroll
    for (int n = 0; n < channels; n++) {
        img[i * width * channels + n] = img[(i * width + 1) * channels + n];
        img[(i * width + (width - 2)) * channels + n] = img[(i * width + (width - 1)) * channels + n];
    }
}

__global__ void copy_all(uchar* img, int width, int height, uchar* out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= height || j >= width) return;

    out[i * width + j] = img[i * width + j];
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

void resize_kernel(uchar* in, int width, int height, int channels, float scale, uchar* out) {
    int new_width = floor(scale * width);
    int new_height = floor(scale * height);

    // Split input into 16x16 (256 threads per grid)
    dim3 grid(new_width / threadBlockWidth + 1, new_height / threadBlockHeight + 1);
    dim3 block(threadBlockWidth, threadBlockHeight);

    cu_resize<<<grid, block>>>(in, width, height, channels, scale, out);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
}

void gaussian3_kernel(uchar* in, int width, int height, uchar* out) {
    // Split input into 16x16 (256 threads per grid)
    dim3 grid(width / threadBlockWidth + 1, height / threadBlockHeight + 1);
    dim3 block(threadBlockWidth, threadBlockHeight);

    cu_gaussian3<<<grid, block>>>(in, width, height, out);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());

    copy_horizontal<<<width - 2, 1>>>(out, width, height, 3);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());

    copy_vertical<<<height, 1>>>(out, width, height, 3);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
}

void gaussian_diff_edge_kernel(uchar* in, int width, int height, uchar* out, uchar* worker_arr, int n_iter, int threshold) {
    uchar* orig_in = in;
    uchar* orig_out = out;
    
    
    dim3 grid(width / threadBlockWidth + 1, height / threadBlockHeight + 1);
    dim3 block(threadBlockWidth, threadBlockHeight);
    
    copy_all<<<grid, block>>>(in, width, height, worker_arr);
    printf("Created in copy\n");
    for (int i = 0; i < n_iter; i++) {
        cu_gaussian<<<grid, block>>>(worker_arr, width, height, out);
        cudaDeviceSynchronize();
        checkCudaCall(cudaGetLastError());
        
        uchar* tmp = out;
        out = worker_arr;
        worker_arr = tmp;
    }
    printf("Smoothed array.\n");

    if (n_iter % 2 == 1) {
        uchar* tmp = out;
        out = worker_arr;
        worker_arr = tmp;
    }

    cu_diff_edge_thresh<<<grid, block>>>(orig_in, worker_arr, width, height, threshold, orig_out);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
}

void dilate_kernel(uchar* in, int width, int height, uchar* out) {
    dim3 grid(width / threadBlockWidth + 1, height / threadBlockHeight + 1);    
    dim3 block(threadBlockWidth, threadBlockHeight);

    cu_dilate<<<grid, block>>>(in, width, height, out);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
}

void luminance_kernel(uchar* in, int width, int height, int channels, uchar* out) {
    // Split input into 16x16 (256 threads per grid)
    dim3 grid(width / threadBlockWidth + 1, height / threadBlockHeight + 1);
    dim3 block(threadBlockWidth, threadBlockHeight);

    cu_luminance<<<grid, block>>>(in, width, height, channels, out);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
}

void push_grad_kernel(uchar* in, int width, int height, uchar* out) {
    // Split input into 16x16 (256 threads per grid)
    dim3 grid(width / threadBlockWidth + 1, height / threadBlockHeight + 1);
    dim3 block(threadBlockWidth, threadBlockHeight);

    cu_push_grad<<<grid, block>>>(in, width, height, out);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());


    copy_horizontal<<<width - 2, 1>>>(out, width, height, 1);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());

    copy_vertical<<<height, 1>>>(out, width, height, 1);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
}


void sobel_kernel(uchar* in, int width, int height, uchar* out) {
    // Split input into 16x16 (256 threads per grid)
    dim3 grid(width / threadBlockWidth + 1, height / threadBlockHeight + 1);
    dim3 block(threadBlockWidth, threadBlockHeight);


    cu_sobel<<<grid, block>>>(in, width, height, out);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());


    copy_horizontal<<<width - 2, 1>>>(out, width, height, 1);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());

    copy_vertical<<<height, 1>>>(out, width, height, 1);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
}

void push_rgb_kernel(uchar* data, uchar* grad, uchar* out, uchar* out_grad, int width, int height, int channels, uchar* bitmask) {
    // Split input into 16x16 (256 threads per grid)
    dim3 grid(width / threadBlockWidth + 1, height / threadBlockHeight + 1);
    dim3 block(threadBlockWidth, threadBlockHeight);

    cu_push_rgb<<<grid, block>>>(data, grad, out, out_grad, width, height, channels, bitmask);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());

    copy_horizontal<<<width - 2, 1>>>(out, width, height, channels);
    cudaDeviceSynchronize();

    copy_vertical<<<height, 1>>>(out, width, height, channels);
    cudaDeviceSynchronize();
    checkCudaCall(cudaGetLastError());
}