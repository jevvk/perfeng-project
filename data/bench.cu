#include <cuda.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>

#define uchar unsigned char
#define RGB_STRENGTH 0.5
#define BITMASK_RADIUS 5

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

__global__ void cu_push_grad(uchar* in, int width, int height, uchar* out, uchar* bitmask) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i == 0 || i >= height - 1 || j == 0 || j >= width - 1) return;

    int c = in[i * width + j];

    if (bitmask[i * width + j] == 0)  {
        out[i * width + j] = c;
        return;
    }

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

void push_grad_kernel(uchar* in, int width, int height, uchar* out, uchar* bitmask) {
    // Split input into 16x16 (256 threads per grid)
    dim3 grid(width / threadBlockWidth + 1, height / threadBlockHeight + 1);
    dim3 block(threadBlockWidth, threadBlockHeight);

    cu_push_grad<<<grid, block>>>(in, width, height, out, bitmask);
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

double get_time(timeval start, timeval end) {
    return (double) (end.tv_usec - start.tv_usec) / 1000000 +
           (double) (end.tv_sec - start.tv_sec);
}

#define ITERS 100

int main() {
    int new_width = 1920 * 2;
    int new_height = 1080 * 2;

    uchar* bitmask = (uchar*) malloc(sizeof(uchar) * new_width * new_height);

    for (int i = 0; i < new_height * new_height; i++) {
        bitmask[i] = 0;
    }

    uchar* remote_grad;
    uchar* remote_rgb;
    uchar* remote_bitmask;
    uchar* remote_x;

    create_device_image((void**) &remote_x,  new_width * new_height * 3 * sizeof(uchar));
    create_device_image((void**) &remote_grad,  new_width * new_height * 3 * sizeof(uchar));
    create_device_image((void**) &remote_rgb, new_width * new_height * sizeof(uchar));
    create_device_image((void**) &remote_bitmask, new_width * new_height * sizeof(uchar));

    copy_to_device(remote_bitmask, bitmask, new_width * new_height * sizeof(uchar));

    struct timeval tv_grad, tv_rgb, tv_end;

    gettimeofday(&tv_grad, NULL);
    for (int i = 0; i < ITERS; i++) {
        push_grad_kernel(remote_grad, new_width, new_height, remote_x, remote_bitmask);
    }
    gettimeofday(&tv_rgb, NULL);
    for (int i = 0; i < ITERS; i++) {
        push_grad_kernel(remote_grad, new_width, new_height, remote_x, remote_bitmask);
    }
    gettimeofday(&tv_end, NULL);

    double t_grad = get_time(tv_grad, tv_rgb) / ITERS * 1000.0;
    double t_rgb = get_time(tv_rgb, tv_end) / ITERS * 1000.0;

    printf("push grad ms:\t%f\n", t_grad);
    printf("push rgb ms:\t%f\n", t_rgb);
    printf("c_copy_grad:\t%f\n", t_grad * 1000.0 / new_width * 1000.0 / new_height);
    printf("c_copy_rgb:\t%f\n", t_rgb * 1000.0 / new_width * 1000.0 / new_height);
}
