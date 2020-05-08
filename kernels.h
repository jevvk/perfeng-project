#ifndef KERNELS_H
#define KERNELS_H

#define uchar unsigned char

#define USE_CUDA

#ifdef __cplusplus
extern "C" {    
}
#endif 

#ifdef USE_CUDA
void create_device_image(void** ptr, int size);

void copy_to_device(void* host, void* device, int size);
void copy_from_device(void* device, void* host, int size);

void resize_kernel(uchar* in, int width, int height, int channels, float scale, uchar* out);
void gaussian3_kernel(uchar* in, int width, int height, uchar* out);

// Our additions.
void gaussian_diff_edge_kernel(uchar* in, int width, int height, uchar* out, uchar* worker_arr, int n_iter, int threshold);
void dilate_kernel(uchar* in, int width, int height, uchar* out);

void luminance_kernel(uchar* in, int width, int height, int channels, uchar* out);
void push_grad_kernel(unsigned char* in, int width, int height, unsigned char* out);
void sobel_kernel(unsigned char* in, int width, int height, unsigned char* out);
void push_rgb_kernel(uchar* data, uchar* grad, uchar* out, uchar* out_grad, int width, int height, int channels, uchar* bitmask);
#else
void resize(uchar* in, int width, int height, int channels, float scale, uchar* out);
void luminance(uchar* in, int width, int height, int channels, uchar* out);
void sobel(uchar* in, int width, int height, uchar* out);
void gaussian(uchar* in, int width, int height, uchar* out);
void gaussian3(uchar* in, int width, int height, uchar* out);
void median(uchar* in, int width, int height, uchar* out);
void median3(uchar* in, int width, int height, uchar* out);
void push_rgb(uchar* data, uchar* grad, uchar* out, uchar* out_grad, int width, int height, int channels);
void push_grad(uchar* in, int width, int height, uchar* out);
#endif

#endif
