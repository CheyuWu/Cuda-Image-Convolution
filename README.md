# Cuda-Image-Convolution# CUDA Image Convolution

This project demonstrates CUDA-based convolution filters (Sobel, Gaussian, etc.) implemented both with custom CUDA kernels and NVIDIA NPP (Performance Primitives).

## Usage

```bash
make
./main images/input.png images/output.png sobel custom
./main images/input.png images/output.png gaussian npp
```

## Project Structure

- include/: Header declarations
- src/: Implementations of CUDA kernels, NPP wrapper, image I/O
- images/: Input/output sample images

## Dependencies
- OpenCV
  ```bash
  $ sudo apt install libopencv-dev
  ```
- CUDA Toolkit (with NPP)
