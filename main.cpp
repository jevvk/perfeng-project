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

#define CHANNELS 3 // bmp always has 3 channels (r, g, b)
#define UNBLUR_ITER 3
#define REFINE_ITER 5
#define BITMASK_DILATE 4
#define GAUSS_ITERS 3
#define THRESHOLD_VAL 1

#define uchar unsigned char

struct cuda_images {
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
};

void usage(char* program_name) {
  printf("Usage: %s scale input_image output_image\n", program_name);
  exit(1);
}

double get_time(timeval start, timeval end) {
  return (double) (end.tv_usec - start.tv_usec) / 1000000 +
         (double) (end.tv_sec - start.tv_sec);
}

double tot_all, tot_res, tot_lum, tot_blur, tot_sobel, tot_refine, tot_end, tot_gaus_diff, tot_transfer;
long long tot_skip;

void init_cuda_images(struct cuda_images* ci, int new_width, int new_height) {
  create_device_image((void**) &ci->remote_original,  new_width * new_height * CHANNELS * sizeof(uchar));
  create_device_image((void**) &ci->remote_blurred,   new_width * new_height * CHANNELS * sizeof(uchar));
  create_device_image((void**) &ci->remote_lum,       new_width * new_height * sizeof(uchar));
  create_device_image((void**) &ci->remote_grad,      new_width * new_height * sizeof(uchar));

  // Data required for possible bitmask optimization.
  create_device_image((void**) &ci->remote_lum_sharp, new_width * new_height * sizeof(uchar));
  create_device_image((void**) &ci->remote_edges,     new_width * new_height * sizeof(uchar));
  create_device_image((void**) &ci->remote_bitmask,   new_width * new_height * sizeof(uchar));

  create_device_image((void**) &ci->remote_tmp1c,     new_width * new_height * sizeof(uchar));
  create_device_image((void**) &ci->remote_res,       new_width * new_height * CHANNELS * sizeof(uchar));
  create_device_image((void**) &ci->remote_tmpnc,     new_width * new_height * CHANNELS * sizeof(uchar));
}

void free_cuda_images(struct cuda_images* ci) {
  free_device_image((void*) ci->remote_original);
  free_device_image((void*) ci->remote_blurred);
  free_device_image((void*) ci->remote_lum);
  free_device_image((void*) ci->remote_grad);

  // Bitmask optimization free
  free_device_image((void*) ci->remote_lum_sharp);
  free_device_image((void*) ci->remote_edges);
  free_device_image((void*) ci->remote_bitmask);

  free_device_image((void*) ci->remote_tmp1c);
  free_device_image((void*) ci->remote_res);
  free_device_image((void*) ci->remote_tmpnc);
}

void process_image_cuda(struct cuda_images* ci, uchar* original, uchar* res, uchar* tmp1c, int width, int height, int scale) {
  void* tmp;

  int new_width = floor(scale * width);
  int new_height = floor(scale * height);

  struct timeval tv_res, tv_lum, tv_blur, tv_sobel, tv_refine, tv_end, tv_gaus_diff, tv_transfer;
  
  gettimeofday(&tv_res, NULL);
  resize_kernel(ci->remote_original, width, height, CHANNELS, scale, ci->remote_res);

  gettimeofday(&tv_lum, NULL);
  luminance_kernel(ci->remote_res, new_width, new_height, CHANNELS, ci->remote_lum_sharp);

  gettimeofday(&tv_gaus_diff, NULL);
  gaussian_diff_edge_kernel(ci->remote_lum_sharp, new_width, new_height, ci->remote_edges, ci->remote_lum, GAUSS_ITERS, THRESHOLD_VAL);

  for (int i = 0; i < BITMASK_DILATE; i++) {
    dilate_kernel(ci->remote_edges, new_width, new_height, ci->remote_tmp1c);

    tmp = ci->remote_edges;
    ci->remote_edges = ci->remote_tmp1c;
    ci->remote_tmp1c = (uchar*) tmp;
  }

  gettimeofday(&tv_blur , NULL);
  for (int i = 0; i < UNBLUR_ITER; i++) {
    push_grad_kernel(ci->remote_lum, new_width, new_height, ci->remote_tmp1c, ci->remote_edges);

    tmp = ci->remote_lum;
    ci->remote_lum = ci->remote_tmp1c;
    ci->remote_tmp1c = (uchar*) tmp;
  }

  gettimeofday(&tv_sobel, NULL);

  sobel_kernel(ci->remote_lum, new_width, new_height, ci->remote_grad);

  gettimeofday(&tv_refine, NULL);
  for (int i = 0; i < REFINE_ITER; i++) {
    push_rgb_kernel(ci->remote_res, ci->remote_grad, ci->remote_tmpnc, ci->remote_tmp1c, new_width, new_height, ci->remote_edges);

    tmp = ci->remote_res;
    ci->remote_res = ci->remote_tmpnc;
    ci->remote_tmpnc = (uchar*) tmp;

    // push_grad(grad, new_width, new_height, tmp1c);

    tmp = ci->remote_grad;
    ci->remote_grad = ci->remote_tmp1c;
    ci->remote_tmp1c = (uchar*) tmp;
  }
  
  gettimeofday(&tv_end, NULL);

  copy_from_device(res, ci->remote_res, new_width * new_height * CHANNELS * sizeof(uchar));
  // copy_from_device(tmp1c, ci->remote_edges, new_width * new_height * sizeof(uchar));

  gettimeofday(&tv_transfer, NULL);
  tot_transfer += get_time(tv_end, tv_transfer);

  // int pos = 0;
  // for (int i = 0; i < new_height; i++) {
  //   for (int j = 0; j < new_width; j++) {
  //     if (tmp1c[i * new_width + j] != 0 && tmp1c[i * new_width + j] != 1)
  //       printf("%d\n", tmp1c[i * new_width + j]);
  //     pos += (tmp1c[i * new_width + j] != 0);
  //   }
  // }
  // tot_skip += pos;

  tot_all += get_time(tv_res, tv_end);
  tot_res += get_time(tv_res, tv_lum);
  tot_lum += get_time(tv_lum, tv_gaus_diff);
  tot_gaus_diff += get_time(tv_gaus_diff, tv_blur);
  tot_blur += get_time(tv_blur, tv_sobel);
  tot_sobel += get_time(tv_sobel, tv_refine);
  tot_refine += get_time(tv_refine, tv_end);
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
  char* input =  (char*) malloc(50*sizeof(char));
  char* output = (char*) malloc(50*sizeof(char));
  unsigned int width, height;
  
  // Set timestamps to 0.
  tot_res = tot_all = tot_lum = tot_blur = tot_sobel = tot_refine = tot_end = tot_gaus_diff = 0;

  // Set skip coutner to 0
  tot_skip = 0;

  struct timeval complete_start, complete_end;
  struct timeval transfer_start, transfer_end;
  gettimeofday(&complete_start, NULL);

  int new_width, new_height;
  int start_idx = 80;
  int end_idx = 81;

  uchar* res;
  uchar* tmp1c;
  struct cuda_images* ci = (struct cuda_images*) malloc(sizeof(struct cuda_images));
  for (int image_idx = start_idx; image_idx < end_idx; image_idx++) {
    sprintf(input, "input/images/%03d.bmp", image_idx);
    sprintf(output, "output/images/%03d.bmp", image_idx);
  

    unsigned int err = loadbmp_decode_file(input, &original, &width, &height, LOADBMP_RGB);
    if (err) {
      printf("Could not open or find the image\n");
      return 1;
    }
    if (image_idx == start_idx) {
      new_width = floor(scale * width);
      new_height = floor(scale * height);
      res = (uchar*) malloc(sizeof(uchar) * new_width * new_height * CHANNELS);
      tmp1c = (uchar*) malloc(sizeof(uchar) * new_width * new_height);
    
      init_cuda_images(ci, new_width, new_height);
    }

    gettimeofday(&transfer_start, NULL);
    copy_to_device(ci->remote_original, original, width * height * CHANNELS * sizeof(uchar));
    // We dont need the original after its copied over.
    free(original);
    gettimeofday(&transfer_end, NULL);
    tot_transfer += get_time(transfer_start, transfer_end);

    // Two arrays to store the resulting image and the bitmask (to count skipped pixels)
#ifdef USE_CUDA
    process_image_cuda(ci, original, res, tmp1c, width, height, scale);
#else
    printf("Not yet implemented...\n");
#endif

    // err = loadbmp_encode_file(output, res, new_width, new_height, LOADBMP_RGB);
    // if (err) {
      // printf("Error during saving file to %s\n", output);
    // }

    if (image_idx == end_idx - 1) {
      free(res);
      free(tmp1c);
      free_cuda_images(ci);
    }
  }
  
  gettimeofday(&complete_end, NULL);
  float tt = get_time(complete_start, complete_end);
  // printf("Overall Time: %.2f\n", tt);
  // printf("Transfer Time: %.2f\n", tot_transfer);
  // printf("---------------\nAlgorithm Stats\n------------------\n");
  // int size = new_width * new_height * (end_idx - start_idx);
  // printf("%lld/%d pixels (%.2f percent skipped)\n", tot_skip, size, (((size - tot_skip)) / (float)size) * 100.0);

  // printf("Total Time:     %.5f (%.2f fps)\n", tot_all, (end_idx - start_idx) / tot_all);
  // printf("  Resizing:     %.5f\n", tot_res);
  // printf("  Luminance:    %.5f\n", tot_lum);
  // printf("  Gauss + Edge: %.5f\n", tot_gaus_diff);
  // printf("  Unblurring:   %.5f\n", tot_blur);
  // printf("  Sobel:        %.5f\n", tot_sobel);
  // printf("  Refining:     %.5f\n", tot_refine);

  float fps = 1 / tot_all;
  printf("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n", tot_all, fps, tot_res, tot_lum, tot_gaus_diff, tot_blur, tot_sobel, tot_refine);

  return 0;
}
