#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>

inline uchar get_pixel(uchar* in, int width, int height, int channels, int row, int col, int channel) {
  return in[row * width * channels + col * channels + channel];
}

void resize(uchar* in, int width, int height, int channels, float scale, uchar* out) {
  int new_width = floor(scale * width);
  int new_height = floor(scale * height);

  for (int i = 0; i < new_height; i++) {
    for (int j = 0; j < new_width; j++) {
      for (int c = 0; c < channels; c++) {
        float sample_row = (float) i * height / new_height;
        float sample_col = (float) j * width / new_width;

        int sample_tl_row = floor(sample_row);
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
  }
}

void luminance(uchar* in, int width, int height, int channels, uchar* out) {
  if (channels != 3 && channels != 1) {
    std::cout << "Unknown image format. It has " << channels << "channels." << std::endl;
    return std::exit(1);
  }

  for (int i = 0; i < height; i++) {
    for (int j = 0; j < width; j++) {
      for (int c = 0; c < channels; c++) {
        if (channels == 3) {
          // int sample = 2 * in[i * width *  channels + j * channels];
          // sample += 3 * in[i * width *  channels + j * channels + 1];
          // sample += in[i * width *  channels + j * channels] + 2;

          float sample = 0.3 * in[i * width * channels + j * channels];
          sample += 0.58 * in[i * width * channels + j * channels + 1];
          sample += 0.11 * in[i * width * channels + j * channels + 2];

          out[i * width + j] = sample;
        } else {
          out[i * width + j] = in[i * width + j];
        }
      }
    }
  }
}

float min(float a, float b) {
  return a > b ? b : a;
}

float max(float a, float b) {
  return a < b ? b : a;
}

float clamp(float val, float min_val, float max_val) {
  return max(min(val, max_val), min_val);
}

void sobel(uchar* in, int width, int height, float* out) {
  float kx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
  float ky[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      float mag_x = 0;
      float mag_y = 0;

      for (int ki = 0; ki < 3; ki++) {
        for (int kj = 0; kj < 3; kj++) {
          mag_x += kx[ki][kj] * in[(i + ki - 1) * width + j + kj - 1];
          mag_y += ky[ki][kj] * in[(i + ki - 1) * width + j + kj - 1];
        }
      }

      out[i * width + j] = 1.0f - clamp(sqrt(mag_x * mag_x + mag_y * mag_y) / 255.0, 0, 1);
    }
  }

  // expand the image by 1px
  // not sure, we could change this

  for (int i = 1; i < width - 1; i ++) {
    out[i] = out[width + i];
    out[(height - 1) * width + i] = out[(height - 2) * width + i];
  }

  for (int i = 1; i < height - 1; i ++) {
    out[i * width] = out[i * width + 1];
    out[(i + 1) * width - 1] = out[(i + 1) * width - 2];
  }

  out[0] = (out[1] + out[width]) / 2;
  out[width - 1] = (out[width - 2] + out[2 * width - 1]) / 2;

  out[(height - 1) * width] = (out[(height - 2) * width] + out[(height - 1) * width + 1]) / 2;
  out[height * width - 1] = (out[height * width - 2] + out[(height - 1) * width - 1]) / 2;
}

void push_lum() {}

#define RGB_STRENGTH 0.4

int argmin(float* grad, int a, int b) {
  return grad[a] > grad[b] ? b : a;
}

int argmax(float* grad, int a, int b) {
  return grad[a] < grad[b] ? b : a;
}

int argmin3(float* grad, int a, int b, int c) {
  return argmin(grad, argmin(grad, a, b), c);
}

int argmax3(float* grad, int a, int b, int c) {
  return argmax(grad, argmax(grad, a, b), c);
}

uchar blend(uchar base, uchar a, uchar b, uchar c) {
  return (1.0 - RGB_STRENGTH) * base + RGB_STRENGTH * (a + b + c) / 3.0;
}

void copyc(uchar* in, int channels, uchar* out, int v) {
  for (int i = 0; i < channels; i++) {
    out[v * channels + i] = in[v * channels + i];
  }
}

void blendc(uchar* in, int channels, uchar* out, int base, int a, int b, int c) {
  for (int i = 0; i < channels; i++) {
    out[base * channels + i] = blend(in[base * channels + i], in[a * channels + i], in[b * channels + i], in[c * channels + i]);
  }
}

void push_rgb(uchar* data, float* grad, uchar* out, int width, int height, int channels) {
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
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
      min = argmin3(grad, tl, t, tr);
      max = argmax3(grad, bl, b, br);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, tl, t, tr);
        continue;
      }

      // vertical push bottom -> top
      min = argmin3(grad, bl, b, br);
      max = argmax3(grad, tl, t, tr);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, bl, b, br);
        continue;
      }

      // horizontal push left -> right
      min = argmin3(grad, tl, l, bl);
      max = argmax3(grad, tr, r, br);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, tl, l, bl);
        continue;
      }

      // horizontal push right -> left
      min = argmin3(grad, tr, r, br);
      max = argmax3(grad, tl, l, bl);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, tr, r, br);
        continue;
      }

      // diagonal push top right -> bottom left
      min = argmin3(grad, t, c, r);
      max = argmax3(grad, l, bl, b);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, t, tr, r);
        continue;
      }

      // diagonal push bottom left -> top right
      min = argmin3(grad, b, c, l);
      max = argmax3(grad, r, tr, t);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, b, bl, l);
        continue;
      }

      // diagonal push top left -> bottom right
      min = argmin3(grad, t, c, l);
      max = argmax3(grad, r, br, b);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, t, tl, l);
        continue;
      }

      // diagonal push bottom right -> top left
      min = argmin3(grad, b, c, r);
      max = argmax3(grad, l, tl, t);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, b, br, r);
        continue;
      }

      copyc(data, channels, out, c);
    }
  }

  for (int i = 0; i < width - 0; i ++) {
    copyc(data, channels, out, i);
    copyc(data, channels, out, (height - 1) * width + i);
  }

  for (int i = 1; i < height - 1; i ++) {
    copyc(data, channels, out, i * width);
    copyc(data, channels, out, (i + 1) * width - 1);
  }
}

void usage(char* program_name) {
  std::cout << "Usage: " << program_name << " scale input_image output_image" << std::endl;
  std::exit(1);
}

int main(int argc, char** argv) {
  if (argc != 4) {
    usage(argv[0]);
  }

  float scale = atof(argv[1]);
  if (scale <= 1.0) {
    std::cout << "Size should be more than 1.0" << std::endl;
    return 1;
  }

  cv::Mat image = cv::imread(argv[2], cv::IMREAD_COLOR);
  if (!image.data) {
    std::cout <<  "Could not open or find the image" << std::endl ;
    return 1;
  }

  cv::Size size = image.size();
  uchar* original = image.data;
  int new_width = floor(scale * size.width);
  int new_height = floor(scale * size.height);
  int channels = image.channels();
  uchar* upscaled = (uchar*) malloc(sizeof(uchar) * new_width * new_height * channels);
  uchar* lum = (uchar*) malloc(sizeof(uchar) * new_width * new_height);
  float* sob = (float*) malloc(sizeof(float) * new_width * new_height);
  uchar* res = (uchar*) malloc(sizeof(uchar) * new_width * new_height * channels);

  std::cout << new_width << " " << new_height << std::endl;

  resize(original, size.width, size.height, channels, scale, upscaled);
  luminance(upscaled, new_width, new_height, channels, lum);
  sobel(lum, new_width, new_height, sob);
  push_rgb(upscaled, sob, res, new_width, new_height, channels);
  
  cv::Mat output(new_height, new_width, CV_8UC3, res);
  cv::imwrite(argv[3], output);

  free(upscaled);

  return 0;
}
