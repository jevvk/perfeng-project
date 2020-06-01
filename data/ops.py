# tot_all, fps, tot_res, tot_lum, tot_gaus_diff, tot_blur, tot_sobel, tot_refine
total_ms = 0.00806
fps = 124.03870
resize_ms = 0.00074
lum_ms = 0.00055
gaus_diff_ms = 0.00256
blur_ms = 0.00063
sobel_ms = 0.00037
refine_ms = 0.00322

inst_integer =  456614740 * 1  # cu_sobel
inst_integer += 230226 * 9     # copy_horizontal
inst_integer += 200094468 * 4  # cu_push_grad
inst_integer += 183439360 * 1  # cu_luminance
inst_integer += 739456700 * 5  # cu_push_rgb
inst_integer += 183246848 * 1  # cu_diff_edge_thresh
inst_integer += 1344153656 * 1 # cu_resize
inst_integer += 129600 * 9     # copy_vertical
inst_integer += 539439984 * 3  # cu_gaussian
inst_integer += 133480448 * 1  # copy_all
inst_integer += 364189416 * 4  # cu_dilate

inst_fp_32 =  8282404 * 1  # cu_sobel
inst_fp_32 += 0 * 9     # copy_horizontal
inst_fp_32 += 0 * 4  # cu_push_grad
inst_fp_32 += 0 * 1  # cu_luminance
inst_fp_32 += 0 * 5  # cu_push_rgb
inst_fp_32 += 0 * 1  # cu_diff_edge_thresh
inst_fp_32 += 398201804 * 1 # cu_resize
inst_fp_32 += 0 * 9     # copy_vertical
inst_fp_32 += 0 * 3  # cu_gaussian
inst_fp_32 += 0 * 1  # copy_all
inst_fp_32 += 0 * 4  # cu_dilate

dram_read_bytes =  8296512 * 1   # cu_sobel
dram_read_bytes += 18609 * 9     # copy_horizontal
dram_read_bytes += 16576949 * 4  # cu_push_grad
dram_read_bytes += 24885024 * 1  # cu_luminance
dram_read_bytes += 41468460 * 5  # cu_push_rgb
dram_read_bytes += 16590112 * 1  # cu_diff_edge_thresh
dram_read_bytes += 6226720 * 1   # cu_resize
dram_read_bytes += 140615 * 9    # copy_vertical
dram_read_bytes += 8297237 * 3   # cu_gaussian
dram_read_bytes += 8295488 * 1   # copy_all
dram_read_bytes += 8265288 * 4   # cu_dilate

dram_write_bytes =  8381888 * 1  # cu_sobel
dram_write_bytes += 26040 * 9    # copy_horizontal
dram_write_bytes += 7959125 * 4  # cu_push_grad
dram_write_bytes += 8745664 * 1  # cu_luminance
dram_write_bytes += 32856403 * 5 # cu_push_rgb
dram_write_bytes += 8625376 * 1  # cu_diff_edge_thresh
dram_write_bytes += 24974240 * 1 # cu_resize
dram_write_bytes += 506833 * 9   # copy_vertical
dram_write_bytes += 7699434 * 3  # cu_gaussian
dram_write_bytes += 8378048 * 1  # copy_all
dram_write_bytes += 8343696 * 4  # cu_dilate

dram_bytes = dram_read_bytes + dram_write_bytes

throughput = dram_bytes / total_ms / (1024 ** 3)
gflops = inst_fp_32 / total_ms / 1e9
giops = inst_integer / total_ms / 1e9

print('GIOPS', giops)
print('GFLOPS', gflops)
print('AI (INT)', giops / throughput)
print('AI (SP)', gflops / throughput)
