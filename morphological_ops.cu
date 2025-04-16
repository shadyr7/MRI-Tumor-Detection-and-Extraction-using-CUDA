// PASTE THE *MODIFIED* morphological_ops.cu CODE HERE
#include <stdexcept> // Required for std::runtime_error
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

// CUDA Kernel for Thresholding (Binary)
__global__ void thresholdKernel(const unsigned char *input, unsigned char *output, int width, int height, unsigned char threshold) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        output[idx] = (input[idx] > threshold) ? 255 : 0;
    }
}

// CUDA Kernel for Dilation with 5x5 structuring element
__global__ void dilationKernel(const unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        unsigned char maxVal = 0; // Dilate looks for max in neighborhood

        // Iterate over the 5x5 neighborhood
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int neighbor_x = x + kx;
                int neighbor_y = y + ky;

                // Check boundaries of the neighbor
                if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                    unsigned char neighbor_val = input[neighbor_y * width + neighbor_x];
                    // Update maxVal if current neighbor is greater
                    if (neighbor_val > maxVal) {
                        maxVal = neighbor_val;
                    }
                    // Optimization: if maxVal is already 255, we can stop searching neighborhood
                    if (maxVal == 255) goto end_dilation_loop;
                }
            }
        }
        end_dilation_loop:; // Label for goto jump

        output[idx] = maxVal;
    }
}

// CUDA Kernel for Erosion with 5x5 structuring element
__global__ void erosionKernel(const unsigned char *input, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        unsigned char minVal = 255; // Erode looks for min in neighborhood

        // Iterate over the 5x5 neighborhood
        for (int ky = -2; ky <= 2; ky++) {
            for (int kx = -2; kx <= 2; kx++) {
                int neighbor_x = x + kx;
                int neighbor_y = y + ky;

                 // Check boundaries of the neighbor
                if (neighbor_x >= 0 && neighbor_x < width && neighbor_y >= 0 && neighbor_y < height) {
                    unsigned char neighbor_val = input[neighbor_y * width + neighbor_x];
                     // Update minVal if current neighbor is smaller
                    if (neighbor_val < minVal) {
                        minVal = neighbor_val;
                    }
                     // Optimization: if minVal is already 0, we can stop searching neighborhood
                     if (minVal == 0) goto end_erosion_loop;
                } else {
                    // Handle border cases for erosion: if neighborhood goes outside,
                    // consider it 0 (minimum possible value) if we are eroding foreground (255)
                    // If we assume the structuring element must be fully contained,
                    // this effectively shrinks the image border.
                    // For simplicity here, we often assume border replication or ignore,
                    // but erosion technically requires minimum, so hitting border implies 0.
                     minVal = 0; // If any part of SE is outside, result is min (0)
                     goto end_erosion_loop;
                }
            }
        }
       end_erosion_loop:; // Label for goto jump

        output[idx] = minVal;
    }
}


// PASTE THIS *ENTIRE main function* into the morphological_ops.cu cell,
// replacing the old main function.

// PASTE THIS *ENTIRE main function* into the morphological_ops.cu cell,
// replacing the old main function.

// PASTE THIS *ENTIRE main function* into the morphological_ops.cu cell,
// replacing the old main function.

int main(int argc, char **argv) {
    // --- Argument Parsing & Initial Checks ---
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <input_edge_image> <output_refined_image> <threshold_val> <dbg_binary_out> <dbg_dilated_out>\n", argv[0]);
        fprintf(stderr, "  threshold_val: Value (0-255) for initial binarization.\n");
        fprintf(stderr, "  dbg_binary_out: Filename to save intermediate binary image.\n");
        fprintf(stderr, "  dbg_dilated_out: Filename to save intermediate dilated image.\n");
        return -1;
    }
    const char* inputFilename = argv[1];
    const char* outputFilename = argv[2];
    int threshold_val = atoi(argv[3]);
    const char* binaryOutputFilename = argv[4];
    const char* dilatedOutputFilename = argv[5];

    if (threshold_val < 0 || threshold_val > 255) {
        fprintf(stderr, "Error: Threshold value must be between 0 and 255. Got %d\n", threshold_val);
        return -1;
    }

    // --- Variable Declarations (Initialize where possible) ---
    cudaEvent_t start = nullptr, stop = nullptr;
    float elapsedTime = 0.0f;
    cudaError_t err = cudaSuccess; // Track status
    Mat edgeImage;
    Mat binaryImage, dilatedImage, refinedEdges;
    int width = 0;
    int height = 0;
    size_t image_size = 0;
    unsigned char *h_edges = nullptr;
    unsigned char *h_binary = nullptr;
    unsigned char *h_dilated = nullptr;
    unsigned char *h_final = nullptr;
    unsigned char *d_edges = nullptr, *d_binary = nullptr, *d_dilated = nullptr, *d_final = nullptr;
    dim3 blockSize(16, 16);
    dim3 gridSize;

    // --- Resource Acquisition & Processing Block ---

    // Create Events
    err = cudaEventCreate(&start); if (err != cudaSuccess) { fprintf(stderr, "cudaEventCreate failed for start: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaEventCreate(&stop); if (err != cudaSuccess) { fprintf(stderr, "cudaEventCreate failed for stop: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaEventRecord(start, 0); if (err != cudaSuccess) { fprintf(stderr, "cudaEventRecord failed for start: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Load Image
    edgeImage = imread(inputFilename, IMREAD_GRAYSCALE);
    if (edgeImage.empty()) {
        fprintf(stderr, "Error: Edge-detected image not found or could not load: %s\n", inputFilename);
        err = cudaErrorFileNotFound; // Indicate error type
        goto cleanup_and_exit;
    }
    if (edgeImage.type() != CV_8UC1) {
        fprintf(stderr, "Error: Input image must be 8-bit 1-channel grayscale (CV_8UC1). Found type %d\n", edgeImage.type());
        err = cudaErrorInvalidValue;
        goto cleanup_and_exit;
    }

    // Get dimensions AFTER successful load
    width = edgeImage.cols;
    height = edgeImage.rows;
    image_size = (size_t)width * height * sizeof(unsigned char);
    printf("Morphological Ops Input: %s (%dx%d), Threshold: %d\n", inputFilename, width, height, threshold_val);

    // Handle image continuity
    h_edges = edgeImage.data;
    if (!edgeImage.isContinuous()) {
        fprintf(stderr, "Warning: Input edge image data is not continuous. Cloning.\n");
        edgeImage = edgeImage.clone();
        h_edges = edgeImage.data;
    }

    // Allocate Host Memory
    h_binary = (unsigned char*)malloc(image_size);
    h_dilated = (unsigned char*)malloc(image_size);
    h_final = (unsigned char*)malloc(image_size);
    if (!h_binary || !h_dilated || !h_final) {
        fprintf(stderr, "Error: Unable to allocate host memory for outputs!\n");
        err = cudaErrorMemoryAllocation;
        goto cleanup_and_exit;
    }

    // Allocate Device Memory
    err = cudaMalloc(&d_edges, image_size); if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for d_edges: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaMalloc(&d_binary, image_size); if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for d_binary: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaMalloc(&d_dilated, image_size); if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for d_dilated: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaMalloc(&d_final, image_size); if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for d_final: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Copy input to device
    err = cudaMemcpy(d_edges, h_edges, image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Calculate Grid Size
    gridSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // --- CUDA Kernels & Intermediate Steps ---

    // 1. Thresholding
    thresholdKernel<<<gridSize, blockSize>>>(d_edges, d_binary, width, height, (unsigned char)threshold_val);
    err = cudaGetLastError(); // Check after kernel
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Kernel Error (thresholdKernel): %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaDeviceSynchronize(); // Check after sync
    if (err != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize after threshold failed: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Save intermediate binary
    err = cudaMemcpy(h_binary, d_binary, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy D2H failed for binary: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    binaryImage = Mat(height, width, CV_8UC1, h_binary);
    if (!imwrite(binaryOutputFilename, binaryImage)) { fprintf(stderr, "Warning: Could not save intermediate binary image %s\n", binaryOutputFilename); }
    else { printf("Morphological Ops Debug: Saved binary image %s\n", binaryOutputFilename); }


    // 2. Dilation
    dilationKernel<<<gridSize, blockSize>>>(d_binary, d_dilated, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Kernel Error (dilationKernel): %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaDeviceSynchronize();
     if (err != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize after dilation failed: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Save intermediate dilated
    err = cudaMemcpy(h_dilated, d_dilated, image_size, cudaMemcpyDeviceToHost);
     if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy D2H failed for dilated: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    dilatedImage = Mat(height, width, CV_8UC1, h_dilated);
    if (!imwrite(dilatedOutputFilename, dilatedImage)) { fprintf(stderr, "Warning: Could not save intermediate dilated image %s\n", dilatedOutputFilename); }
    else { printf("Morphological Ops Debug: Saved dilated image %s\n", dilatedOutputFilename); }


    // 3. Erosion
    erosionKernel<<<gridSize, blockSize>>>(d_dilated, d_final, width, height);
    err = cudaGetLastError();
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Kernel Error (erosionKernel): %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaDeviceSynchronize();
     if (err != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize after erosion failed: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // --- Timing & Final Output ---
    // Only time if everything up to here succeeded
    err = cudaEventRecord(stop, 0); if (err != cudaSuccess) { fprintf(stderr, "cudaEventRecord failed for stop: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaEventSynchronize(stop); if (err != cudaSuccess) { fprintf(stderr, "cudaEventSynchronize failed for stop: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaEventElapsedTime(&elapsedTime, start, stop); if (err != cudaSuccess) { fprintf(stderr, "cudaEventElapsedTime failed: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Copy final result back
    err = cudaMemcpy(h_final, d_final, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy D2H failed for final: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Save final image
    refinedEdges = Mat(height, width, CV_8UC1, h_final);
    if (!imwrite(outputFilename, refinedEdges)) { fprintf(stderr, "Error: Could not save refined edges image %s\n", outputFilename); /* Potentially non-fatal? */ }
    else { printf("Morphological Ops Output: %s\n", outputFilename); }

cleanup_and_exit:
    // --- Unified Cleanup ---
    if (d_edges) cudaFree(d_edges);
    if (d_binary) cudaFree(d_binary);
    if (d_dilated) cudaFree(d_dilated);
    if (d_final) cudaFree(d_final);

    free(h_binary);
    free(h_dilated);
    free(h_final);

    if(start) cudaEventDestroy(start);
    if(stop) cudaEventDestroy(stop);

    if (err == cudaSuccess && elapsedTime > 0) {
        printf("Morphological Ops Time: %.3f ms\n", elapsedTime);
    } else if (err != cudaSuccess) {
         printf("Morphological Ops finished with error: %s\n", cudaGetErrorString(err));
    }

    return (err == cudaSuccess) ? 0 : -1;
}
