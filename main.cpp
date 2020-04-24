#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <math.h>

#define RGB_STRENGTH 0.5
#define UNBLUR_ITER 3
#define REFINE_ITER 5

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
          // int sample = in[i * width *  channels + j * channels];
          // sample += 3 * in[i * width *  channels + j * channels + 1];
          // sample += 2* in[i * width *  channels + j * channels] + 2;

          float sample = 0.11 * in[i * width * channels + j * channels];
          sample += 0.58 * in[i * width * channels + j * channels + 1];
          sample += 0.3 * in[i * width * channels + j * channels + 2];

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

uchar min(uchar a, uchar b) {
  return a > b ? b : a;
}

uchar max(uchar a, uchar b) {
  return a < b ? b : a;
}

uchar min3(uchar a, uchar b, uchar c) {
  return min(min(a, b), c);
}

uchar max3(uchar a, uchar b, uchar c) {
  return max(max(a, b), c);
}

void sobel(uchar* in, int width, int height, uchar* out) {
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

      out[i * width + j] = 255.0 - clamp(sqrt(mag_x * mag_x + mag_y * mag_y), 0, 255.0);
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

void gaussian(uchar* in, int width, int height, uchar* out) {
  float g[3][3] = {{1/16.0, 1/8.0, 1/16.0}, {1/8.0, 1/4.0, 1/8.0}, {1/16.0, 1/8.0, 1/16.0}};

  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      float res = 0;

      for (int ki = 0; ki < 3; ki++) {
        for (int kj = 0; kj < 3; kj++) {
          res += g[ki][kj] * in[(i + ki - 1) * width + j + kj - 1];
        }
      }

      out[i * width + j] = res;
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

void copyc(uchar* in, int channels, uchar* out, int v) {
  for (int i = 0; i < channels; i++) {
    out[v * channels + i] = in[v * channels + i];
  }
}

void gaussian3(uchar* in, int width, int height, uchar* out) {
  float g[3][3] = {{1/16.0, 1/8.0, 1/16.0}, {1/8.0, 1/4.0, 1/8.0}, {1/16.0, 1/8.0, 1/16.0}};

  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
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
  }

  // expand the image by 1px
  // not sure, we could change this

  for (int i = 1; i < width - 1; i ++) {
    copyc(in, 3, out, i);
    copyc(in, 3, out, (height - 1) * width + i);
  }

  for (int i = 0; i < height; i ++) {
    copyc(in, 3, out, i * width);
    copyc(in, 3, out, (i + 1) * width - 1);
  }
}

void median(uchar* in, int width, int height, uchar* out) {
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      uchar vals[9];
      uchar* cur = vals;

      for (int ki = -1; ki < 2; ki++) {
        for (int kj = -1; kj < 2; kj++) {
          *cur = in[(i + ki) * width + j + kj];
          cur++;
        }
      }

      int len = 9;
      int swap = 1;
      int tmp, new_len;
      while (swap) {
        swap = 0;

        for (int i = 1; i < len; i++) {
          if (vals[i - 1] > vals[i]) {
            swap = 1;
            tmp = vals[i];
            vals[i] = vals[i - 1];
            vals[i - 1] = tmp;
            new_len = i;
          }
        }

        len = new_len;
      }

      // for (int i = 0; i < 9; i++) printf("%d ", vals[i]);
      // printf("\n");

      out[i * width + j] = vals[5];
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

void bgr2hsv(uchar* bgr, float* hsv) {
  float M = (float) max3(bgr[0], bgr[1], bgr[2]) / 255.0;
  float m = (float) min3(bgr[0], bgr[1], bgr[2]) / 255.0;
  float C = M - m;

  hsv[2] = M;

  if (C <= 0.0001f) {
    hsv[0] = 0;
    hsv[1] = 0; // technically undefined
    return;
  }

  if (M == 0) {
    hsv[0] = 0;
    hsv[1] = 0; // technically undefined
    return;
  }

  hsv[1] = C / M;

  if (bgr[2] >= M * 255.0) {
    hsv[0] = ((float) bgr[1] - bgr[0]) / 255.0 / C;
  } else if (bgr[1] >= M * 255.0) {
    hsv[0] = 2.0 + ((float) bgr[0] - bgr[2]) / 255.0 / C;
  } else {
    hsv[0] = 4.0 + ((float) bgr[2] - bgr[1]) / 255.0 / C;
  }

  hsv[0] *= 60;

  if (hsv[0] < 0.0) {
    hsv[0] += 360.0;
  }
}

void hsv2bgr(float* hsv, uchar* bgr) {
  if (hsv[1] <= 0) {
    bgr[0] = hsv[2] * 255;
    bgr[1] = hsv[2] * 255;
    bgr[2] = hsv[2] * 255;
    return;
  }

  float hh, p, q, t, ff;
  long i;

  hh = hsv[0];

  if (hh >= 360.0) {
    hh = 0.0;
  }

  hh /= 60.0;
  i = (long) hh;
  ff = hh - i;
  p = hsv[2] * (1.0 - hsv[1]);
  q = hsv[2] * (1.0 - (hsv[1] * ff));
  t = hsv[2] * (1.0 - (hsv[1] * (1.0 - ff)));

  switch(i) {
  case 0:
    bgr[2] = hsv[2] * 255;
    bgr[1] = t * 255;
    bgr[0] = p * 255;
    return;

  case 1:
    bgr[2] = q * 255;
    bgr[1] = hsv[2] * 255;
    bgr[0] = p * 255;
    return;

  case 2:
    bgr[2] = p * 255;
    bgr[1] = hsv[2] * 255;
    bgr[0] = t * 255;
    return;

  case 3:
    bgr[2] = p * 255;
    bgr[1] = q * 255;
    bgr[0] = hsv[2] * 255;
    return;

  case 4:
    bgr[2] = t * 255;
    bgr[1] = p * 255;
    bgr[0] = hsv[2] * 255;
    return;

  case 5:
  default:
    bgr[2] = hsv[2] * 255;
    bgr[1] = p * 255;
    bgr[0] = q * 255;
    return;
  }
}

void median3(uchar* in, int width, int height, uchar* out) {
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
      uchar bgr[9][3];
      float hsv[9][3];
      uchar med_bgr[3];
      float med_hsv[3];
      
      for (int ki = -1, k = 0; ki < 2; ki++) {
        for (int kj = -1; kj < 2; kj++) {
          bgr[k][0] = in[(i + ki) * width * 3 + (j + kj) * 3];
          bgr[k][1] = in[(i + ki) * width * 3 + (j + kj) * 3 + 1];
          bgr[k][2] = in[(i + ki) * width * 3 + (j + kj) * 3 + 2];

          k++;
        }
      }

      for (int k = 0; k < 9; k++) {
        bgr2hsv(bgr[k], hsv[k]);
      }

      for (int c = 0; c < 3; c++) {
        float vals[9];

        for (int k = 0; k < 9; k++) {
          vals[k] = hsv[k][c];
        }

        int len = 9;
        int swap = 1;
        int new_len;
        float tmp;
        while (swap) {
          swap = 0;

          for (int i = 1; i < len; i++) {
            if (vals[i - 1] > vals[i]) {
              swap = 1;
              tmp = vals[i];
              vals[i] = vals[i - 1];
              vals[i - 1] = tmp;
              new_len = i;
            }
          }

          len = new_len;
        }

        med_hsv[c] = vals[5];
      }

      hsv2bgr(med_hsv, med_bgr);

      out[i * width * 3 + j * 3] = med_bgr[0];
      out[i * width * 3 + j * 3 + 1] = med_bgr[1];
      out[i * width * 3 + j * 3 + 2] = med_bgr[2];
    }
  }

  // expand the image by 1px
  // not sure, we could change this

  for (int i = 1; i < width - 1; i ++) {
    copyc(in, 3, out, i);
    copyc(in, 3, out, (height - 1) * width + i);
  }

  for (int i = 0; i < height; i ++) {
    copyc(in, 3, out, i * width);
    copyc(in, 3, out, (i + 1) * width - 1);
  }
}

int argmin(uchar* grad, int a, int b) {
  return grad[a] > grad[b] ? b : a;
}

int argmax(uchar* grad, int a, int b) {
  return grad[a] < grad[b] ? b : a;
}

int argmin3(uchar* grad, int a, int b, int c) {
  return argmin(grad, argmin(grad, a, b), c);
}

int argmax3(uchar* grad, int a, int b, int c) {
  return argmax(grad, argmax(grad, a, b), c);
}

uchar blend(uchar base, uchar a, uchar b, uchar c) {
  return (1.0 - RGB_STRENGTH) * base + RGB_STRENGTH * (a + b + c) / 3.0;
}

void blendc(uchar* in, int channels, uchar* out, int base, int a, int b, int c) {
  for (int i = 0; i < channels; i++) {
    out[base * channels + i] = blend(in[base * channels + i], in[a * channels + i], in[b * channels + i], in[c * channels + i]);
  }
}

void push_rgb(uchar* data, uchar* grad, uchar* out, uchar* out_grad, int width, int height, int channels) {
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

      if (grad[min] > grad[max]) {
        blendc(data, channels, out, c, tl, t, tr);
        out_grad[c] = blend(grad[c], grad[tl], grad[t], grad[tr]);
        continue;
      }

      // vertical push bottom -> top
      min = argmin3(grad, bl, b, br);
      max = argmax3(grad, tl, t, tr);

      if (grad[min] > grad[max]) {
        blendc(data, channels, out, c, bl, b, br);
        out_grad[c] = blend(grad[c], grad[bl], grad[b], grad[br]);
        continue;
      }

      // horizontal push left -> right
      min = argmin3(grad, tl, l, bl);
      max = argmax3(grad, tr, r, br);

      if (grad[min] > grad[max]) {
        blendc(data, channels, out, c, tl, l, bl);
        out_grad[c] = blend(grad[c], grad[tl], grad[l], grad[bl]);
        continue;
      }

      // horizontal push right -> left
      min = argmin3(grad, tr, r, br);
      max = argmax3(grad, tl, l, bl);

      if (grad[min] > grad[max]) {
        blendc(data, channels, out, c, tr, r, br);
        out_grad[c] = blend(grad[c], grad[tr], grad[r], grad[br]);
        continue;
      }

      // diagonal push top right -> bottom left
      min = argmin3(grad, t, c, r);
      max = argmax3(grad, l, bl, b);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, t, tr, r);
        out_grad[c] = blend(grad[c], grad[t], grad[tr], grad[r]);
        continue;
      }

      // diagonal push bottom left -> top right
      min = argmin3(grad, b, c, l);
      max = argmax3(grad, r, tr, t);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, b, bl, l);
        out_grad[c] = blend(grad[c], grad[b], grad[bl], grad[l]);
        continue;
      }

      // diagonal push top left -> bottom right
      min = argmin3(grad, t, c, l);
      max = argmax3(grad, r, br, b);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, t, tl, l);
        out_grad[c] = blend(grad[c], grad[t], grad[tl], grad[l]);
        continue;
      }

      // diagonal push bottom right -> top left
      min = argmin3(grad, b, c, r);
      max = argmax3(grad, l, tl, t);

      if (grad[min] > grad[c] && grad[c] > grad[max]) {
        blendc(data, channels, out, c, b, br, r);
        out_grad[c] = blend(grad[c], grad[b], grad[br], grad[r]);
        continue;
      }

      copyc(data, channels, out, c);
      out_grad[c] = grad[c];
    }
  }

  for (int i = 1; i < width - 1; i ++) {
    copyc(data, channels, out, i);
    copyc(data, channels, out, (height - 1) * width + i);
  }

  for (int i = 0; i < height; i ++) {
    copyc(data, channels, out, i * width);
    copyc(data, channels, out, (i + 1) * width - 1);
  }
}

uchar blend_lightest(uchar lightest, uchar base, uchar a, uchar b, uchar c) {
  return max(lightest, blend(base, a, b, c));
}

void push_grad(uchar* in, int width, int height, uchar* out) {
  for (int i = 1; i < height - 1; i++) {
    for (int j = 1; j < width - 1; j++) {
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
      min = min3(tl, t, tr);
      max = max3(bl, b, br);

      uchar res = c;

      if (min > max) {
        res = blend_lightest(res, c, tl, t, tr);
      }

      // vertical push bottom -> top
      min = min3(bl, b, br);
      max = max3(tl, t, tr);

      if (min > max) {
        res = blend_lightest(res, c, bl, b, br);
      }

      // horizontal push left -> right
      min = min3(tl, l, bl);
      max = max3(tr, r, br);

      if (min > max) {
        res = blend_lightest(res, c, tl, l, bl);
      }

      // horizontal push right -> left
      min = min3(tr, r, br);
      max = max3(tl, l, bl);

      if (min > max) {
        res = blend_lightest(res, c, tr, r, br);
      }

      // diagonal push top right -> bottom left
      min = min3(t, c, r);
      max = max3(l, bl, b);

      if (min > res && res > max) {
        res = blend_lightest(res, c, t, tr, r);
      }

      // diagonal push bottom left -> top right
      min = min3(b, c, l);
      max = max3(r, tr, t);

      if (min > res && res > max) {
        res = blend_lightest(res, c, b, bl, l);
      }

      // diagonal push top left -> bottom right
      min = min3(t, c, l);
      max = max3(r, br, b);

      if (min > res && res > max) {
        res = blend_lightest(res, c, t, tl, l);
      }

      // diagonal push bottom right -> top left
      min = min3(b, c, r);
      max = max3(l, tl, t);

      if (min > res && res > max) {
        res = blend_lightest(res, c, b, br, r);
      }

      out[i * width + j] = res;
    }
  }

  // expand the image by 1px
  // not sure, we could change this

  for (int i = 1; i < width - 1; i ++) {
    out[i] = out[width + i];
    out[(height - 1) * width + i] = out[(height - 2) * width + i];
  }

  for (int i = 0; i < height; i ++) {
    out[i * width] = out[i * width + 1];
    out[(i + 1) * width - 1] = out[(i + 1) * width - 2];
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

  void* tmp;
  uchar* tmp1c = (uchar*) malloc(sizeof(uchar) * new_width * new_height);
  uchar* tmpnc = (uchar*) malloc(sizeof(uchar) * new_width * new_height * channels);

  uchar* upscaled = (uchar*) malloc(sizeof(uchar) * new_width * new_height * channels);
  uchar* blurred = (uchar*) malloc(sizeof(uchar) * new_width * new_height * channels);
  uchar* lum = (uchar*) malloc(sizeof(uchar) * new_width * new_height);
  uchar* med = (uchar*) malloc(sizeof(uchar) * new_width * new_height);
  uchar* sob = (uchar*) malloc(sizeof(float) * new_width * new_height);
  uchar* grad = (uchar*) malloc(sizeof(float) * new_width * new_height);
  uchar* res = (uchar*) malloc(sizeof(uchar) * new_width * new_height * channels);

  std::cout << new_width << " " << new_height << std::endl;

  // median3(original, size.width, size.height, blurred);
  // resize(blurred, size.width, size.height, channels, scale, res);
  // luminance(res, new_width, new_height, channels, lum);
  resize(original, size.width, size.height, channels, scale, res);
  median3(res, new_width, new_height, blurred);
  // repeatedly applying median filter seems to shift the colors
  // median3(blurred, new_width, new_height, res);
  // median3(res, new_width, new_height, blurred);
  luminance(blurred, new_width, new_height, channels, lum);
  // luminance(res, new_width, new_height, channels, lum);

  for (int i = 0; i < UNBLUR_ITER; i++) {
    push_grad(lum, new_width, new_height, tmp1c);

    tmp = lum;
    lum = tmp1c;
    tmp1c = (uchar*) tmp;
  }

  sobel(lum, new_width, new_height, grad);

  for (int i = 0; i < REFINE_ITER; i++) {
    push_rgb(res, grad, tmpnc, tmp1c, new_width, new_height, channels);

    tmp = res;
    res = tmpnc;
    tmpnc = (uchar*) tmp;

    // push_grad(grad, new_width, new_height, tmp1c);

    tmp = grad;
    grad = tmp1c;
    tmp1c = (uchar*) tmp;
  }
  
  // cv::Mat output(new_height, new_width, CV_8U, grad);
  cv::Mat output(new_height, new_width, CV_8UC3, res);
  cv::imwrite(argv[3], output);

  free(upscaled);
  free(lum);
  free(med);
  free(sob);
  free(res);

  return 0;
}
