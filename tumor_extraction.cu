// PASTE THE *MODIFIED* tumor_extraction.cu CODE HERE
#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include <stdio.h> // Use stdio for fprintf, printf
#include <string>

using namespace cv;
// No need for iostream if using printf/fprintf
// using namespace std;

// CUDA error-checking macro
#define CHECK_CUDA(call) do {                                  \
    cudaError_t err = call;                                    \
    if (err != cudaSuccess) {                                  \
        fprintf(stderr, "CUDA Error at %s:%d - %s\n",          \
               __FILE__, __LINE__, cudaGetErrorString(err));   \
        exit(EXIT_FAILURE);                                    \
    }                                                          \
} while(0)


// CUDA Kernel to Apply Mask (Extract Only Tumor)
// Input: original grayscale image, binary mask, output buffer
__global__ void applyMaskKernel(const unsigned char *image, const unsigned char *mask, unsigned char *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * width + x;

    // Boundary check
    if (x < width && y < height) {
        // Check the mask value for the current pixel
        // Treat any non-zero value in the mask as part of the tumor region
        bool isTumor = (mask[idx] > 0); // Simpler check: 0 is background, >0 is foreground

        // If it's part of the tumor (mask > 0), copy the original image pixel value.
        // Otherwise, set the output pixel to 0 (black).
        output[idx] = isTumor ? image[idx] : 0;
    }
}


int main(int argc, char **argv) {
     if (argc != 4) {
        fprintf(stderr, "Usage: %s <input_grayscale_image> <input_tumor_mask> <output_extracted_tumor>\n", argv[0]);
        return -1;
    }
    const char* brainImageFilename = argv[1];
    const char* tumorMaskFilename = argv[2];
    const char* outputFilename = argv[3];

    // Load the grayscale MRI and tumor mask images
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, 0));

    Mat brainImage = imread(brainImageFilename, IMREAD_GRAYSCALE);
    Mat tumorMask = imread(tumorMaskFilename, IMREAD_GRAYSCALE);

    if (brainImage.empty()) {
        fprintf(stderr, "Error: Unable to load grayscale brain image: %s\n", brainImageFilename);
        return -1;
    }
     if (tumorMask.empty()) {
        fprintf(stderr, "Error: Unable to load tumor mask image: %s\n", tumorMaskFilename);
        return -1;
    }

    // Ensure both images have the same dimensions
    if (brainImage.size() != tumorMask.size()) {
        fprintf(stderr, "Error: Image (%dx%d) and mask (%dx%d) size mismatch!\n",
                brainImage.cols, brainImage.rows, tumorMask.cols, tumorMask.rows);
        return -1;
    }
    // Ensure images are grayscale
     if (brainImage.type() != CV_8UC1) {
         fprintf(stderr, "Error: Brain image must be 8-bit 1-channel grayscale (CV_8UC1). Found type %d\n", brainImage.type());
         return -1;
     }
      if (tumorMask.type() != CV_8UC1) {
         // Try to convert mask if it's not grayscale (e.g., loaded as color accidentally)
         fprintf(stderr, "Warning: Tumor mask is not grayscale (type %d). Attempting conversion.\n", tumorMask.type());
         cvtColor(tumorMask, tumorMask, COLOR_BGR2GRAY); // Assume BGR if not grayscale
          if (tumorMask.type() != CV_8UC1) {
             fprintf(stderr, "Error: Could not convert tumor mask to grayscale.\n");
             return -1;
          }
     }


    int width = brainImage.cols;
    int height = brainImage.rows;
    size_t image_size = (size_t)width * height * sizeof(unsigned char);

    printf("Tumor Extraction Input: Image=%s, Mask=%s (%dx%d)\n", brainImageFilename, tumorMaskFilename, width, height);


    // Prepare host data pointers (ensure continuity if needed)
    unsigned char *h_brain = brainImage.data;
    unsigned char *h_mask = tumorMask.data;
    if (!brainImage.isContinuous()) {
        fprintf(stderr, "Warning: Brain image data not continuous. Cloning.\n");
        brainImage = brainImage.clone();
        h_brain = brainImage.data;
    }
     if (!tumorMask.isContinuous()) {
        fprintf(stderr, "Warning: Mask image data not continuous. Cloning.\n");
        tumorMask = tumorMask.clone();
        h_mask = tumorMask.data;
    }

    // Allocate host memory for the output
    unsigned char *h_output = (unsigned char*)malloc(image_size);
     if (!h_output) {
        fprintf(stderr, "Error: Unable to allocate host memory for extracted tumor output!\n");
        return -1;
    }


    // Allocate memory on the device
    unsigned char *d_brain, *d_mask, *d_output;
    CHECK_CUDA(cudaMalloc(&d_brain, image_size));
    CHECK_CUDA(cudaMalloc(&d_mask, image_size));
    CHECK_CUDA(cudaMalloc(&d_output, image_size));

    // Copy images to the device
    CHECK_CUDA(cudaMemcpy(d_brain, h_brain, image_size, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_mask, h_mask, image_size, cudaMemcpyHostToDevice));
    // It's good practice to initialize output memory if the kernel doesn't write every pixel
    // In this case, it does (either image value or 0), so memset isn't strictly needed.
    // CHECK_CUDA(cudaMemset(d_output, 0, image_size));


    // Define CUDA grid and block sizes
    dim3 blockSize(16, 16); // 256 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);


    // Launch CUDA kernel to extract the tumor
    applyMaskKernel<<<gridSize, blockSize>>>(d_brain, d_mask, d_output, width, height);

     // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Kernel Error (applyMaskKernel): %s\n", cudaGetErrorString(err));
        // Cleanup before exit
        cudaFree(d_brain); cudaFree(d_mask); cudaFree(d_output);
        free(h_output);
        return -1;
    }
    // Synchronize device to ensure kernel completion and get accurate timing
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float elapsedTime;
    CHECK_CUDA(cudaEventElapsedTime(&elapsedTime, start, stop));


    // Retrieve the extracted tumor image from device to host memory
    CHECK_CUDA(cudaMemcpy(h_output, d_output, image_size, cudaMemcpyDeviceToHost));

    // Save the extracted tumor image using OpenCV
    Mat extractedTumor(height, width, CV_8UC1, h_output); // Create Mat header for host data
     if (!imwrite(outputFilename, extractedTumor)) {
         fprintf(stderr, "Error: Could not save extracted tumor image to %s\n", outputFilename);
     } else {
         printf("Tumor Extraction Output: %s\n", outputFilename);
     }


    // Free Memory (Device and Host)
    CHECK_CUDA(cudaFree(d_brain));
    CHECK_CUDA(cudaFree(d_mask));
    CHECK_CUDA(cudaFree(d_output));
    free(h_output); // Free host memory allocated with malloc
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    printf("Tumor Extraction Time: %.3f ms\n", elapsedTime);

    return 0; // Success
}
