
// (The version that takes input/output filenames as arguments)
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string> // For string comparison

using namespace cv;

// Macro for checking CUDA errors
#define CHECK_CUDA(call) do {                                  \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n",          \
               __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while(0)

// CUDA Kernel for RGB to Grayscale
__global__ void rgbToGrayKernel(unsigned char *rgb, unsigned char *gray, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check is crucial for safety
    if (x < width && y < height) {
        int rgb_idx = (y * width + x) * 3;
        int gray_idx = y * width + x;
        // Check bounds for rgb access as well, though less likely if width/height are correct
        // This simple check assumes width*height*3 doesn't overflow, which is reasonable for images
        // A more robust check would involve calculating max index based on total allocation size

        // Using integer math to avoid float conversions in kernel if possible
        unsigned int gray_val = (299 * (unsigned int)rgb[rgb_idx] +
                                 587 * (unsigned int)rgb[rgb_idx + 1] +
                                 114 * (unsigned int)rgb[rgb_idx + 2]) / 1000;

        // Clamp value to 0-255 range, although the weights should prevent overflow > 255
        gray[gray_idx] = (unsigned char)(gray_val > 255 ? 255 : gray_val);
    }
}


int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_rgb_image> <output_gray_image>\n", argv[0]);
        return -1;
    }
    const char* inputFilename = argv[1];
    const char* outputFilename = argv[2];

    // Load Image
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    Mat image = imread(inputFilename, IMREAD_COLOR);
    if (image.empty()) {
        fprintf(stderr, "Error: Image not found or could not be loaded: %s\n", inputFilename);
        return -1;
    }
    // Ensure image is 3-channel unsigned char
    if (image.type() != CV_8UC3) {
         fprintf(stderr, "Error: Input image must be 8-bit 3-channel (CV_8UC3). Found type %d\n", image.type());
         // Attempt conversion if possible, or exit
         if(image.channels() == 1) {
            fprintf(stderr, "Input is already grayscale? Converting to BGR for consistency, but this might not be intended.\n");
            cvtColor(image, image, COLOR_GRAY2BGR);
         } else if (image.channels() == 4) {
            fprintf(stderr, "Input has 4 channels (e.g., BGRA), converting to BGR.\n");
            cvtColor(image, image, COLOR_BGRA2BGR);
         } else {
            fprintf(stderr, "Cannot handle input image type %d.\n", image.type());
            return -1;
         }
         if (image.type() != CV_8UC3) { // Check again after potential conversion
              fprintf(stderr, "Error: Could not convert image to CV_8UC3.\n");
              return -1;
         }
    }


    int width = image.cols;
    int height = image.rows;
    size_t rgb_size = (size_t)width * height * 3 * sizeof(unsigned char);
    size_t gray_size = (size_t)width * height * sizeof(unsigned char);

    printf("Grayscale Input: %s (%dx%d)\n", inputFilename, width, height);

    // Host data pointers
    unsigned char *h_rgb = image.data; // Use image data directly if continuous
    if (!image.isContinuous()) {
        fprintf(stderr, "Warning: Image data is not continuous. Making a clone.\n");
        image = image.clone(); // Make it continuous
        h_rgb = image.data;
    }
    unsigned char *h_gray = (unsigned char*)malloc(gray_size);
    if (!h_gray) {
        fprintf(stderr, "Error: Unable to allocate host memory for grayscale output!\n");
        return -1;
    }

    // Allocate Memory on Device
    unsigned char *d_rgb, *d_gray;
    CHECK_CUDA(cudaMalloc(&d_rgb, rgb_size));
    CHECK_CUDA(cudaMalloc(&d_gray, gray_size));

    // Copy input image data to device
    CHECK_CUDA(cudaMemcpy(d_rgb, h_rgb, rgb_size, cudaMemcpyHostToDevice));

    // Define CUDA Grid
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Run Kernel
    rgbToGrayKernel<<<gridSize, blockSize>>>(d_rgb, d_gray, width, height);

    // Check for CUDA kernel execution errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (rgbToGrayKernel): %s\n", cudaGetErrorString(err));
        // Cleanup before exit
        cudaFree(d_rgb);
        cudaFree(d_gray);
        free(h_gray);
        return -1;
    }
    // Ensure kernel is finished before stopping timer and proceeding
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));

    // Copy back to host
    CHECK_CUDA(cudaMemcpy(h_gray, d_gray, gray_size, cudaMemcpyDeviceToHost));

    // Save the grayscale image
    Mat grayImage(height, width, CV_8UC1, h_gray); // Create Mat header for host data
    if (!imwrite(outputFilename, grayImage)) {
         fprintf(stderr, "Error: Could not save grayscale image to %s\n", outputFilename);
         // Continue cleanup, but report error
    } else {
         printf("Grayscale Output: %s\n", outputFilename);
    }


    // Free Memory
    CHECK_CUDA(cudaFree(d_rgb));
    CHECK_CUDA(cudaFree(d_gray));
    free(h_gray); // Free host memory allocated with malloc
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));


    printf("Grayscale Time: %.3f ms\n", elapsedTime);
    return 0;
}
