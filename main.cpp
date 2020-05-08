// #include <opencv2/core/core.hpp>
// #include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
// #include <iostream>
#include <math.h>
#include <stdlib.h>
#include <sys/time.h>
#include <stdio.h>
#include "kernels.h"

#define LOADBMP_IMPLEMENTATION
#include "bmp.h"

#define RGB_STRENGTH 0.5
#define UNBLUR_ITER 3
#define REFINE_ITER 5
#define BITMASK_DILATE 3
#define GAUSS_ITERS 3
#define THRESHOLD_VAL 1

#define uchar unsigned char

void usage(char* program_name) {
  printf("Usage: %s scale input_image output_image\n", program_name);
  exit(1);
}

double get_time(timeval start, timeval end) {
  return (double) (end.tv_usec - start.tv_usec) / 1000000 +
         (double) (end.tv_sec - start.tv_sec);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    usage(argv[0]);
  }

  float scale = atof(argv[1]);
  if (scale <= 1.0) {
    printf("Size should be more than 1.0\n");
    return 1;
  }

  uchar *original;
  unsigned int width, height;

  unsigned int err = loadbmp_decode_file(argv[2], &original, &width, &height, LOADBMP_RGB);
  if (err) {
    printf("Could not open or find the image\n");
    return 1;
  }

  int new_width = floor(scale * width);
  int new_height = floor(scale * height);
  int channels = 3; // BMP always has 3 channels (r, g, b)

  void* tmp;
  uchar* tmp1c = (uchar*) malloc(sizeof(uchar) * new_width * new_height);
  uchar* tmpnc = (uchar*) malloc(sizeof(uchar) * new_width * new_height * channels);

  uchar* upscaled = (uchar*) malloc(sizeof(uchar) * new_width * new_height * channels);
  uchar* blurred = (uchar*) malloc(sizeof(uchar) * new_width * new_height * channels);
  uchar* lum = (uchar*) malloc(sizeof(uchar) * new_width * new_height);
  uchar* med = (uchar*) malloc(sizeof(uchar) * new_width * new_height);
  uchar* sob = (uchar*) malloc(sizeof(uchar) * new_width * new_height);
  uchar* grad = (uchar*) malloc(sizeof(uchar) * new_width * new_height);
  uchar* res = (uchar*) malloc(sizeof(uchar) * new_width * new_height * channels);

  /*
   * GPU Allocations and memcpys
   */

  uchar* remote_lum;
  uchar* remote_grad;
  uchar* remote_res;
  uchar* remote_tmpnc;
  uchar* remote_tmp1c;
  uchar* remote_blurred;
  uchar* remote_original;
  uchar* remote_lum_sharp;
  uchar* remote_edges;
  uchar* remote_bitmask;

  create_device_image((void**) &remote_original,  new_width * new_height * channels * sizeof(uchar));
  create_device_image((void**) &remote_blurred,  new_width * new_height * channels * sizeof(uchar));

  create_device_image((void**) &remote_lum,       new_width * new_height * sizeof(uchar));
  create_device_image((void**) &remote_grad,      new_width * new_height * sizeof(uchar));

  // Data required for possible bitmask optimization.
  create_device_image((void**) &remote_lum_sharp, new_width * new_height * sizeof(uchar));
  create_device_image((void**) &remote_edges,     new_width * new_height * sizeof(uchar));
  create_device_image((void**) &remote_bitmask,   new_width * new_height * sizeof(uchar));

  create_device_image((void**) &remote_tmp1c,  new_width * new_height * sizeof(uchar));
  
  create_device_image((void**) &remote_res,    new_width * new_height * channels * sizeof(uchar));
  create_device_image((void**) &remote_tmpnc,  new_width * new_height * channels * sizeof(uchar));

  copy_to_device(remote_original, original, width * height * channels * sizeof(uchar));

  printf("Creating image of %dx%d\n", new_width, new_height);

  struct timeval tv_res, tv_med, tv_lum, tv_blur, tv_sobel, tv_refine, tv_end;
  struct timeval tv_gaus_diff;
  
  gettimeofday(&tv_res, NULL);
  resize_kernel(remote_original, width, height, channels, scale, remote_res);

  gettimeofday(&tv_med, NULL);
  // gaussian3_kernel(remote_res, new_width, new_height, remote_blurred);

  // TODO: Optimize this to not do double luminance and double gaussian.
  gettimeofday(&tv_lum, NULL);
  // luminance_kernel(remote_blurred, new_width, new_height, channels, remote_lum);
  luminance_kernel(remote_res, new_width, new_height, channels, remote_lum_sharp);

  gettimeofday(&tv_gaus_diff, NULL);
  gaussian_diff_edge_kernel(remote_lum_sharp, new_width, new_height, remote_edges, remote_lum, GAUSS_ITERS, THRESHOLD_VAL);

  for (int i = 0; i < BITMASK_DILATE; i++) {
    dilate_kernel(remote_edges, new_width, new_height, remote_tmp1c);

    tmp = remote_edges;
    remote_edges = remote_tmp1c;
    remote_tmp1c = (uchar*) tmp;
  }

  gettimeofday(&tv_blur , NULL);
  for (int i = 0; i < UNBLUR_ITER; i++) {
    push_grad_kernel(remote_lum, new_width, new_height, remote_tmp1c, remote_edges);

    tmp = remote_lum;
    remote_lum = remote_tmp1c;
    remote_tmp1c = (uchar*) tmp;
  }

  gettimeofday(&tv_sobel, NULL);

  sobel_kernel(remote_lum, new_width, new_height, remote_grad);

  gettimeofday(&tv_refine, NULL);
  for (int i = 0; i < REFINE_ITER; i++) {
    push_rgb_kernel(remote_res, remote_grad, remote_tmpnc, remote_tmp1c, new_width, new_height, channels, remote_edges);

    tmp = remote_res;
    remote_res = remote_tmpnc;
    remote_tmpnc = (uchar*) tmp;

    // push_grad(grad, new_width, new_height, tmp1c);

    tmp = remote_grad;
    remote_grad = remote_tmp1c;
    remote_tmp1c = (uchar*) tmp;
  }
  
  gettimeofday(&tv_end, NULL);
  
  copy_from_device(res, remote_res, new_width * new_height * channels * sizeof(uchar));
  copy_from_device(tmp1c, remote_edges, new_width * new_height * sizeof(uchar));

  err = loadbmp_encode_file(argv[3], res, new_width, new_height, LOADBMP_RGB);
  if (err) {
    printf("Error during saving file to %s\n", argv[3]);
  }

  int pos = 0;
  for (int i = 0; i < new_height; i++) {
    for (int j = 0; j < new_width; j++) {
      pos += tmp1c[i * new_width + j];
    }
  }
  int size = new_width * new_height;
  printf("%d/%d pixels (%.2f percent skipped)\n", pos, size, (((size - pos)) / (float)size) * 100.0);


  free(upscaled);
  free(lum);
  free(med);
  free(sob);
  free(res);

  printf("Total compute time: %.5f\n", get_time(tv_res, tv_end));
  printf("  Resizing:   %.5f\n", get_time(tv_res, tv_med));
  printf("  Blurring:   %.5f\n", get_time(tv_med, tv_lum));
  printf("  Luminance:  %.5f\n", get_time(tv_lum, tv_gaus_diff));
  printf("  Gauss + Edge:  %.5f\n", get_time(tv_gaus_diff, tv_blur));
  printf("  Unblurring: %.5f\n", get_time(tv_blur, tv_sobel));
  printf("  Sobel:      %.5f\n", get_time(tv_sobel, tv_refine));
  printf("  Refining:   %.5f\n", get_time(tv_refine, tv_end));

  return 0;
}
