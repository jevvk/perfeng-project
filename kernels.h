#ifndef KERNELS_H
#define KERNELS_H


#ifdef __cplusplus
extern "C" {    
}
#endif 

void create_device_image(void** ptr, int size);

void copy_to_device(void* host, void* device, int size);
void copy_from_device(void* device, void* host, int size);

void sobel_kernel(unsigned char* in, int width, int height, unsigned char* out);

#endif
