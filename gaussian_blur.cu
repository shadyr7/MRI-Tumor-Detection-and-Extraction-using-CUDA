// PASTE THE *MODIFIED* gaussian_blur.cu CODE HERE
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <string>

using namespace cv;

// CUDA error-checking macro
#define CHECK_CUDA(call) do {                                  \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n",          \
               __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while(0)

// Gaussian Kernel (3x3) - Remains in constant memory
__constant__ float d_kernel[9] = {
    0.0625f, 0.125f, 0.0625f,
    0.125f,  0.25f,  0.125f,
    0.0625f, 0.125f, 0.0625f
};

// CUDA Kernel for Gaussian Blur
__global__ void gaussianBlurKernel(const unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Check boundary conditions first
    if (x >= width || y >= height) return;

    // Handle border pixels: copy original value (simplest approach)
    // More complex handling (e.g., replicate, reflect) could be implemented
    if (x == 0 || y == 0 || x == width - 1 || y == height - 1) {
        output[y * width + x] = input[y * width + x];
    } else {
        // Apply 3x3 Gaussian filter for inner pixels
        float sum = 0.0f;
        int kernel_idx = 0; // Index for the 1D constant kernel array

        for (int ky = -1; ky <= 1; ky++) {
            for (int kx = -1; kx <= 1; kx++) {
                // Calculate index for the neighboring pixel in the input image
                // Note: No need for boundary checks here because we already handled borders
                int neighbor_idx = (y + ky) * width + (x + kx);

                // Accumulate weighted sum
                sum += (float)input[neighbor_idx] * d_kernel[kernel_idx++];
            }
        }
        // Clamp the result to the valid range [0, 255] and cast to unsigned char
        output[y * width + x] = (unsigned char)fminf(fmaxf(sum, 0.0f), 255.0f);
    }
}


int main(int argc, char **argv) {
     if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_gray_image> <output_blurred_image>\n", argv[0]);
        return -1;
    }
    const char* inputFilename = argv[1];
    const char* outputFilename = argv[2];

    // Load Grayscale Image
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    Mat grayImage = imread(inputFilename, IMREAD_GRAYSCALE);
    if (grayImage.empty()) {
        fprintf(stderr, "Error: Grayscale image not found or could not load: %s\n", inputFilename);
        return -1;
    }
     if (grayImage.type() != CV_8UC1) {
         fprintf(stderr, "Error: Input image must be 8-bit 1-channel grayscale (CV_8UC1). Found type %d\n", grayImage.type());
         return -1;
     }


    int width = grayImage.cols;
    int height = grayImage.rows;
    size_t image_size = (size_t)width * height * sizeof(unsigned char);

    printf("Gaussian Blur Input: %s (%dx%d)\n", inputFilename, width, height);


    // Use image data directly if continuous, otherwise clone
    unsigned char *h_gray = grayImage.data;
     if (!grayImage.isContinuous()) {
        fprintf(stderr, "Warning: Input grayscale image data is not continuous. Cloning.\n");
        grayImage = grayImage.clone();
        h_gray = grayImage.data;
    }

    // Allocate Host Memory for output
    unsigned char *h_blurred = (unsigned char*)malloc(image_size);
    if (!h_blurred) {
        fprintf(stderr, "Error: Unable to allocate host memory for blurred output!\n");
        return -1;
    }

    // Allocate Device Memory
    unsigned char *d_gray, *d_blurred;
    CHECK_CUDA(cudaMalloc(&d_gray, image_size));
    CHECK_CUDA(cudaMalloc(&d_blurred, image_size));

    // Copy input image to device
    CHECK_CUDA(cudaMemcpy(d_gray, h_gray, image_size, cudaMemcpyHostToDevice));

    // Define CUDA Grid
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Run Gaussian Blur Kernel
    gaussianBlurKernel<<<gridSize, blockSize>>>(d_gray, d_blurred, width, height);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (gaussianBlurKernel): %s\n", cudaGetErrorString(err));
        // Cleanup before exit
        cudaFree(d_gray);
        cudaFree(d_blurred);
        free(h_blurred);
        return -1;
    }
     // Synchronize device to ensure kernel completion and get accurate timing
    CHECK_CUDA(cudaDeviceSynchronize());


    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));


    // Copy back the blurred image
    CHECK_CUDA(cudaMemcpy(h_blurred, d_blurred, image_size, cudaMemcpyDeviceToHost));

    // Save the blurred image
    Mat blurredImage(height, width, CV_8UC1, h_blurred); // Create Mat header for host data
     if (!imwrite(outputFilename, blurredImage)) {
         fprintf(stderr, "Error: Could not save blurred image to %s\n", outputFilename);
     } else {
          printf("Gaussian Blur Output: %s\n", outputFilename);
     }


    // Free Memory
    CHECK_CUDA(cudaFree(d_gray));
    CHECK_CUDA(cudaFree(d_blurred));
    free(h_blurred); // Free host memory allocated with malloc
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("Gaussian Blur Time: %.3f ms\n", elapsedTime);
    return 0;
}
