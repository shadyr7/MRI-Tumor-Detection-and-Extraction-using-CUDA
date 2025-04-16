// PASTE THE *MODIFIED* edge_detection.cu CODE HERE
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <math.h> // For sqrtf
#include <string>

using namespace cv;

#define CHECK_CUDA(call) do {                                  \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n",          \
               __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while(0)

// Sobel kernels (can be implicitly used in calculation)
// Gx = [-1 0 1]   Gy = [-1 -2 -1]
//      [-2 0 2]        [ 0  0  0]
//      [-1 0 1]        [ 1  2  1]

__global__ void sobelEdgeDetectionKernel(const unsigned char *blurred, unsigned char *edges, int width, int height, int threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary check: Only compute Sobel for inner pixels
    if (x > 0 && y > 0 && x < width - 1 && y < height - 1) {
        // Calculate Gx (horizontal gradient)
        int Gx = (-1 * blurred[(y-1)*width + (x-1)]) + (0 * blurred[(y-1)*width + (x)]) + (1 * blurred[(y-1)*width + (x+1)]) +
                 (-2 * blurred[(y)  *width + (x-1)]) + (0 * blurred[(y)  *width + (x)]) + (2 * blurred[(y)  *width + (x+1)]) +
                 (-1 * blurred[(y+1)*width + (x-1)]) + (0 * blurred[(y+1)*width + (x)]) + (1 * blurred[(y+1)*width + (x+1)]);

        // Calculate Gy (vertical gradient)
        int Gy = (-1 * blurred[(y-1)*width + (x-1)]) + (-2 * blurred[(y-1)*width + (x)]) + (-1 * blurred[(y-1)*width + (x+1)]) +
                 (0 * blurred[(y)  *width + (x-1)]) + (0 * blurred[(y)  *width + (x)]) + (0 * blurred[(y)  *width + (x+1)]) +
                 (1 * blurred[(y+1)*width + (x-1)]) + (2 * blurred[(y+1)*width + (x)]) + (1 * blurred[(y+1)*width + (x+1)]);

        // Calculate gradient magnitude: sqrt(Gx^2 + Gy^2)
        // Use sqrtf for float sqrt. Cast Gx, Gy to float for intermediate calc if they might be large.
        // Alternatively, approximate magnitude with abs(Gx) + abs(Gy) for speed if precision isn't critical.
        float magnitude_f = sqrtf((float)(Gx * Gx) + (float)(Gy * Gy));

        // Clamp magnitude to [0, 255] and apply threshold
        int magnitude = (int)fminf(fmaxf(magnitude_f, 0.0f), 255.0f);
        edges[y * width + x] = (magnitude > threshold) ? 255 : 0;

    } else {
        // Set border pixels to 0 (black)
        // This check ensures we don't write out of bounds if x or y are exactly width/height
         if (x < width && y < height) {
            edges[y * width + x] = 0;
         }
    }
}

int main(int argc, char **argv) {
    if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_blurred_image> <output_edge_image> <threshold>\n", argv[0]);
        return -1;
    }
    const char* inputFilename = argv[1];
    const char* outputFilename = argv[2];
    int threshold = atoi(argv[3]); // Convert threshold string to integer

    if (threshold < 0 || threshold > 255) {
         fprintf(stderr, "Error: Threshold must be between 0 and 255 (inclusive). Got %d\n", threshold);
         return -1;
    }

    // Load Blurred Grayscale Image
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    Mat blurredImage = imread(inputFilename, IMREAD_GRAYSCALE);
    if (blurredImage.empty()) {
        fprintf(stderr, "Error: Blurred image not found or could not load: %s\n", inputFilename);
        return -1;
    }
     if (blurredImage.type() != CV_8UC1) {
         fprintf(stderr, "Error: Input image must be 8-bit 1-channel grayscale (CV_8UC1). Found type %d\n", blurredImage.type());
         return -1;
     }

    int width = blurredImage.cols;
    int height = blurredImage.rows;
    size_t image_size = (size_t)width * height * sizeof(unsigned char);

    printf("Edge Detection Input: %s (%dx%d), Threshold: %d\n", inputFilename, width, height, threshold);

    // Use image data directly if continuous, otherwise clone
    unsigned char *h_blurred = blurredImage.data;
    if (!blurredImage.isContinuous()) {
        fprintf(stderr, "Warning: Input blurred image data is not continuous. Cloning.\n");
        blurredImage = blurredImage.clone();
        h_blurred = blurredImage.data;
    }

    // Allocate Host Memory for output
    unsigned char *h_edges = (unsigned char*)malloc(image_size);
    if (!h_edges) {
        fprintf(stderr, "Error: Unable to allocate host memory for edge output!\n");
        return -1;
    }

    // Allocate Device Memory
    unsigned char *d_blurred, *d_edges;
    CHECK_CUDA(cudaMalloc(&d_blurred, image_size));
    CHECK_CUDA(cudaMalloc(&d_edges, image_size));

    // Copy input image to device
    CHECK_CUDA(cudaMemcpy(d_blurred, h_blurred, image_size, cudaMemcpyHostToDevice));
    // Initialize edges device memory to 0 (important for border pixels)
    CHECK_CUDA(cudaMemset(d_edges, 0, image_size));

    // Define CUDA Grid
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);

    // Run Sobel Edge Detection Kernel
    sobelEdgeDetectionKernel<<<gridSize, blockSize>>>(d_blurred, d_edges, width, height, threshold);

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (sobelEdgeDetectionKernel): %s\n", cudaGetErrorString(err));
        // Cleanup before exit
        cudaFree(d_blurred);
        cudaFree(d_edges);
        free(h_edges);
        return -1;
    }
    // Synchronize device to ensure kernel completion and get accurate timing
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));

    // Copy back the edge-detected image
    CHECK_CUDA(cudaMemcpy(h_edges, d_edges, image_size, cudaMemcpyDeviceToHost));

    // Save Edge-detected Image
    Mat edgeImage(height, width, CV_8UC1, h_edges); // Create Mat header for host data
     if (!imwrite(outputFilename, edgeImage)) {
         fprintf(stderr, "Error: Could not save edge image to %s\n", outputFilename);
     } else {
          printf("Edge Detection Output: %s\n", outputFilename);
     }

    // Free Memory
    CHECK_CUDA(cudaFree(d_blurred));
    CHECK_CUDA(cudaFree(d_edges));
    free(h_edges); // Free host memory allocated with malloc
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("Edge Detection Time: %.3f ms\n", elapsedTime);
    return 0;
}
