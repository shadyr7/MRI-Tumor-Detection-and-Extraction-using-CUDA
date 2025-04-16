// PASTE THE *MODIFIED* tumor_detection.cu CODE HERE
#include <stdio.h>
#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>
#include <vector>
#include <numeric> // for std::accumulate
#include <limits>  // for numeric_limits

using namespace cv;
using namespace std;

#define CHECK_CUDA(call) do {                                  \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n",          \
               __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while(0)


// --- Host function to calculate Otsu's threshold ---
// This is typically done on the CPU as it requires histogram calculation.
int calculateOtsuThreshold(const Mat& grayImage) {
    if (grayImage.empty() || grayImage.channels() != 1) {
        fprintf(stderr, "Error (Otsu): Input image must be single-channel grayscale.\n");
        return -1; // Indicate error
    }

    const int histSize = 256; // 0-255 levels
    float range[] = { 0, 256 }; // Upper bound is exclusive
    const float* histRange = { range };
    bool uniform = true;
    bool accumulate = false;
    Mat hist;

    // Calculate histogram
    calcHist(&grayImage, 1, 0, Mat(), hist, 1, &histSize, &histRange, uniform, accumulate);

    // Normalize histogram (sum to 1) - Optional but good practice for probabilities
    // double totalPixels = grayImage.rows * grayImage.cols;
    // hist /= totalPixels; // Or calculate probabilities directly below

    float totalPixels = (float)(grayImage.rows * grayImage.cols);
    vector<float> probabilities(histSize);
    float totalMean = 0.0f; // Mean intensity of the whole image
    for(int i = 0; i < histSize; ++i) {
        probabilities[i] = hist.at<float>(i) / totalPixels;
        totalMean += i * probabilities[i];
    }

    float maxVariance = 0.0f;
    int optimalThreshold = 0;
    float P1 = 0.0f; // Cumulative probability of class 1 (background)
    float M1 = 0.0f; // Cumulative mean of class 1

    for (int t = 0; t < histSize; ++t) {
        P1 += probabilities[t]; // Probability of background up to threshold t
        M1 += t * probabilities[t]; // Weighted sum for background mean calculation

        if (P1 == 0.0f || P1 == 1.0f) { // Avoid division by zero or trivial split
            continue;
        }

        float P2 = 1.0f - P1; // Probability of class 2 (foreground)
        float M2 = (totalMean - M1) / P2; // Mean of foreground (derived from total mean)

        float mean1 = M1 / P1; // Mean intensity of class 1

        // Between-class variance: sigma_b^2 = P1 * P2 * (mean1 - mean2)^2
        float variance = P1 * P2 * (mean1 - M2) * (mean1 - M2);

        if (variance > maxVariance) {
            maxVariance = variance;
            optimalThreshold = t;
        }
    }
     printf("Otsu calculated threshold: %d\n", optimalThreshold);
    return optimalThreshold;
}


// --- CUDA Kernels --- (Potentially less effective for simple thresholding/growing than OpenCV CPU)

// Kernel: Simple Thresholding (can use Otsu value calculated on host)
__global__ void thresholdBinaryKernel(const unsigned char *input, unsigned char *output, int width, int height, unsigned char thresholdValue) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x < width && y < height) {
        output[idx] = (input[idx] > thresholdValue) ? 255 : 0;
    }
}


// Note: A simple region growing like this in CUDA without proper synchronization or
// more complex algorithm (like connected components) can be tricky to get right and
// might not be faster than optimized CPU versions (like OpenCV's findContours or floodFill)
// for complex shapes. This version is a basic seed fill spreader.
__global__ void simpleRegionGrowIter(const unsigned char* current_mask, unsigned char* next_mask, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    if (x >= width || y >= height) return;

    // If the current pixel is part of the region (255)
    if (current_mask[idx] == 255) {
         // Mark this pixel in the next mask as well
        next_mask[idx] = 255;

        // Try to expand to direct neighbors (4-connectivity) if they are within bounds
        // Check neighbor pixel *value* in the original image (or thresholded) if needed,
        // but here we assume we just grow based on connectivity if a seed exists.
        // The atomicMax ensures that if multiple threads try to write 255, it happens safely.
        // We are writing to the *next* mask based on the *current* one.

        // Check left neighbor
        if (x > 0 && current_mask[idx - 1] != 255) { // Only expand if neighbor isn't already set
            atomicMax((unsigned int*)&next_mask[idx - 1], (unsigned int)255); // Requires cast for atomic
        }
         // Check right neighbor
         if (x < width - 1 && current_mask[idx + 1] != 255) {
            atomicMax((unsigned int*)&next_mask[idx + 1], (unsigned int)255);
        }
         // Check top neighbor
         if (y > 0 && current_mask[idx - width] != 255) {
            atomicMax((unsigned int*)&next_mask[idx - width], (unsigned int)255);
        }
         // Check bottom neighbor
         if (y < height - 1 && current_mask[idx + width] != 255) {
            atomicMax((unsigned int*)&next_mask[idx + width], (unsigned int)255);
        }
        // 8-connectivity could be added here similarly
    } else {
         // If current pixel is not 255, it might become 255 in the next iteration
         // Ensure it starts as 0 in the next mask if not written by atomicMax
         // This requires initializing next_mask to 0 before the kernel launch or careful handling.
         // A simpler approach: copy current_mask to next_mask first, then grow.
    }
}

// PASTE THIS *ENTIRE main function* into the tumor_detection.cu cell,
// replacing the old main function.

int main(int argc, char **argv) {
     if (argc != 3) {
        fprintf(stderr, "Usage: %s <input_refined_edge_image> <output_tumor_mask>\n", argv[0]);
        return -1;
    }
    const char* inputFilename = argv[1];
    const char* outputFilename = argv[2];

    // --- Variable Declarations ---
    cudaEvent_t start = nullptr, stop = nullptr;
    float elapsedTime = 0.0f;
    cudaError_t err = cudaSuccess;
    Mat refinedEdgeImage;
    Mat tumorMask; // <<< Declare Mat object early
    int width = 0;
    int height = 0;
    size_t image_size = 0;
    unsigned char *h_output_mask = nullptr;
    unsigned char *d_input = nullptr, *d_thresholded = nullptr;
    dim3 blockSize(16, 16);
    dim3 gridSize;
    int otsuThreshold = 0;

    // --- Resource Acquisition & Processing ---

    err = cudaEventCreate(&start); if (err != cudaSuccess) { fprintf(stderr, "cudaEventCreate failed for start: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaEventCreate(&stop); if (err != cudaSuccess) { fprintf(stderr, "cudaEventCreate failed for stop: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaEventRecord(start, 0); if (err != cudaSuccess) { fprintf(stderr, "cudaEventRecord failed for start: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Load Image
    refinedEdgeImage = imread(inputFilename, IMREAD_GRAYSCALE);
    if (refinedEdgeImage.empty()) {
        fprintf(stderr, "Error: Refined edges image not found or could not load: %s\n", inputFilename);
        err = cudaErrorFileNotFound;
        goto cleanup_and_exit;
    }
    if (refinedEdgeImage.type() != CV_8UC1) {
         fprintf(stderr, "Error: Input image must be 8-bit 1-channel grayscale (CV_8UC1). Found type %d\n", refinedEdgeImage.type());
         err = cudaErrorInvalidValue;
         goto cleanup_and_exit;
     }

    // Get Dimensions
    width = refinedEdgeImage.cols;
    height = refinedEdgeImage.rows;
    image_size = (size_t)width * height * sizeof(unsigned char);
    printf("Tumor Detection Input: %s (%dx%d)\n", inputFilename, width, height);

    // --- Step 1: Otsu's Thresholding (Calculated on CPU) ---
    otsuThreshold = calculateOtsuThreshold(refinedEdgeImage);
    if (otsuThreshold < 0) {
        fprintf(stderr, "Error calculating Otsu's threshold.\n");
        err = cudaErrorInvalidValue; // Or a custom error code
        goto cleanup_and_exit;
    }
    if (otsuThreshold == 0) { printf("Warning: Otsu threshold is 0. Image might be mostly black.\n"); }
    else if (otsuThreshold == 255) { printf("Warning: Otsu threshold is 255. Image might be mostly white.\n"); }

    // Allocate Host Memory
    h_output_mask = (unsigned char*)malloc(image_size);
    if (!h_output_mask) {
        fprintf(stderr, "Error: Unable to allocate host memory for output mask!\n");
        err = cudaErrorMemoryAllocation;
        goto cleanup_and_exit;
    }

    // Allocate Device Memory
    err = cudaMalloc(&d_input, image_size); if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for d_input: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaMalloc(&d_thresholded, image_size); if (err != cudaSuccess) { fprintf(stderr, "cudaMalloc failed for d_thresholded: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Copy input image to device
    err = cudaMemcpy(d_input, refinedEdgeImage.data, image_size, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy H2D failed: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Calculate Grid Size
    gridSize = dim3((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // --- Step 2: Apply Otsu Threshold using CUDA Kernel ---
    thresholdBinaryKernel<<<gridSize, blockSize>>>(d_input, d_thresholded, width, height, (unsigned char)otsuThreshold);
    err = cudaGetLastError(); // Check after kernel
    if (err != cudaSuccess) { fprintf(stderr, "CUDA Kernel Error (thresholdBinaryKernel): %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; } // <<< This goto caused the original error
    err = cudaDeviceSynchronize(); // Check after sync
    if (err != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize after threshold failed: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // --- Step 3: Use thresholded image as mask ---
    err = cudaMemcpy(h_output_mask, d_thresholded, image_size, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) { fprintf(stderr, "cudaMemcpy D2H failed for mask: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Record timing
    err = cudaEventRecord(stop, 0); if (err != cudaSuccess) { fprintf(stderr, "cudaEventRecord failed for stop: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaEventSynchronize(stop); if (err != cudaSuccess) { fprintf(stderr, "cudaEventSynchronize failed for stop: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }
    err = cudaEventElapsedTime(&elapsedTime, start, stop); if (err != cudaSuccess) { fprintf(stderr, "cudaEventElapsedTime failed: %s\n", cudaGetErrorString(err)); goto cleanup_and_exit; }

    // Save the final tumor mask
    // <<< Assign to pre-declared Mat object here
    tumorMask = Mat(height, width, CV_8UC1, h_output_mask);
    if (!imwrite(outputFilename, tumorMask)) {
         fprintf(stderr, "Error: Could not save tumor mask image to %s\n", outputFilename);
         // Decide if this is fatal, maybe set err? For now, just print warning.
    } else {
        printf("Tumor Detection Output: %s\n", outputFilename);
    }

cleanup_and_exit:
    // --- Unified Cleanup ---
    if (d_input) cudaFree(d_input);
    if (d_thresholded) cudaFree(d_thresholded);
    free(h_output_mask); // free(nullptr) is safe

    if(start) cudaEventDestroy(start);
    if(stop) cudaEventDestroy(stop);

    if (err == cudaSuccess && elapsedTime > 0) {
        printf("Tumor Detection Time (GPU part): %.3f ms\n", elapsedTime);
    } else if (err != cudaSuccess) {
         printf("Tumor Detection finished with error: %s\n", cudaGetErrorString(err));
    }

    return (err == cudaSuccess) ? 0 : -1;
}
