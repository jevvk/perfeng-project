#ifndef KERNELS_H
#define KERNELS_H


#ifdef __cplusplus
extern "C" {    
}
#endif 

void create_device_image(unsigned char* ptr, int size);

void copy_to_device(unsigned char* host, unsigned char* device, int size);
void copy_from_device(unsigned char* device, unsigned char* host, int size);

void sobel_kernel(unsigned char* in, int width, int height, unsigned char* out);

#endif
