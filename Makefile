INCLUDES=-I/usr/include/opencv4
LIBRARIES=-lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_imgproc

comic-upscaler: main.cpp
	g++ $(INCLUDES) $(LIBRARIES) $^ -o $@

clean:
	rm comic-upscaler
