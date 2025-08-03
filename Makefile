# Simple Makefile for CUDA + OpenCV + NPP project
NVCC=nvcc
CXX=g++
INCLUDES=`pkg-config --cflags opencv4` -I./include -I/usr/local/cuda/include
LIBS=`pkg-config --libs opencv4` -L/usr/local/cuda/lib64 \
  -lnppc -lnppial -lnppicc -lnppig -lnppisu

SRCS=src/main.cpp src/convolution_kernels.cu src/npp_convolution_wrapper.cpp src/image_utils.cpp

main: $(SRCS)
	$(NVCC) $(SRCS) -o main $(INCLUDES) $(LIBS)

clean:
	rm -f main *.o
