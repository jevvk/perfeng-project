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
          int sample = 2 * in[i * width *  channels + j * channels];
          sample += 3 * in[i * width *  channels + j * channels + 1];
          sample += in[i * width *  channels + j * channels] + 2;

          out[i * width + j] = sample / 6;
        } else {
          out[i * width + j] = in[i * width + j];
        }
      }
    }
  }
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

      out[i * width + j] = sqrt(mag_x * mag_x + mag_y * mag_y);
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

void push_rgb() {}
void push_lum() {}

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
  uchar* upscaled = (uchar*) malloc(sizeof(uchar) * new_width * new_height * image.channels());
  uchar* lum = (uchar*) malloc(sizeof(uchar) * new_width * new_height);
  uchar* sob = (uchar*) malloc(sizeof(uchar) * new_width * new_height);

  std::cout << new_width << " " << new_height << std::endl;

  resize(original, size.width, size.height, image.channels(), scale, upscaled);
  luminance(upscaled, new_width, new_height, image.channels(), lum);
  sobel(lum, new_width, new_height, sob);
  
  cv::Mat output(new_height, new_width, CV_8U, sob);
  cv::imwrite(argv[3], output);

  free(upscaled);

  return 0;
}
