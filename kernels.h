#ifndef KERNELS_H
#define KERNELS_H

#define uchar unsigned char

#ifdef __cplusplus
extern "C" {    
}
#endif 

void create_device_image(void** ptr, int size);

void copy_to_device(void* host, void* device, int size);
void copy_from_device(void* device, void* host, int size);

void gaussian3_kernel(uchar* in, int width, int height, uchar* out);
void luminance_kernel(uchar* in, int width, int height, int channels, uchar* out);
void push_grad_kernel(unsigned char* in, int width, int height, unsigned char* out);
void sobel_kernel(unsigned char* in, int width, int height, unsigned char* out);
void push_rgb_kernel(uchar* data, uchar* grad, uchar* out, uchar* out_grad, int width, int height, int channels);

#endif
