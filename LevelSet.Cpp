#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define NX 128          // Grid size in X
#define NY 128          // Grid size in Y
#define NZ 128          // Grid size in Z
#define BLOCK_SIZE 8    // CUDA block size
#define DX 0.01f        // Grid spacing
#define DT 0.0001f      // Time step
#define STEPS 1000      // Number of time steps

// CUDA error checking macro
#define checkCuda(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code: %d)\n", cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    } \
}

// WENO reconstruction for derivatives (1D implementation for simplicity)
__device__ float weno_derivative(float v1, float v2, float v3, float v4, float v5) {
    float eps = 1e-6f; // Regularization parameter

    // Smoothness indicators
    float beta1 = (13.0f/12.0f) * (v1 - 2*v2 + v3)*(v1 - 2*v2 + v3) + (1.0f/4.0f) * (v1 - 4*v2 + 3*v3)*(v1 - 4*v2 + 3*v3);
    float beta2 = (13.0f/12.0f) * (v2 - 2*v3 + v4)*(v2 - 2*v3 + v4) + (1.0f/4.0f) * (v2 - v4)*(v2 - v4);
    float beta3 = (13.0f/12.0f) * (v3 - 2*v4 + v5)*(v3 - 2*v4 + v5) + (1.0f/4.0f) * (3*v3 - 4*v4 + v5)*(3*v3 - 4*v4 + v5);

    // Weights
    float alpha1 = 0.1f / ((beta1 + eps) * (beta1 + eps));
    float alpha2 = 0.6f / ((beta2 + eps) * (beta2 + eps));
    float alpha3 = 0.3f / ((beta3 + eps) * (beta3 + eps));

    float sum_alpha = alpha1 + alpha2 + alpha3;
    float w1 = alpha1 / sum_alpha;
    float w2 = alpha2 / sum_alpha;
    float w3 = alpha3 / sum_alpha;

    // Numerical flux
    float flux = w1 * (2*v1 - 7*v2 + 11*v3) / 6.0f
               + w2 * (-v2 + 5*v3 + 2*v4) / 6.0f
               + w3 * (2*v3 + 5*v4 - v5) / 6.0f;

    return flux / DX;
}

// Level set evolution kernel
__global__ void level_set_kernel(float* phi, float* phi_new) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i >= 2 && i < NX-2 && j >= 2 && j < NY-2 && k >= 2 && k < NZ-2) {
        int idx = (k * NY + j) * NX + i;

        // WENO derivatives in x, y, z directions
        float dphi_dx = weno_derivative(phi[idx-2], phi[idx-1], phi[idx], phi[idx+1], phi[idx+2]);
        float dphi_dy = weno_derivative(phi[idx-2*NX], phi[idx-NX], phi[idx], phi[idx+NX], phi[idx+2*NX]);
        float dphi_dz = weno_derivative(phi[idx-2*NX*NY], phi[idx-NX*NY], phi[idx], phi[idx+NX*NY], phi[idx+2*NX*NY]);

        // Upwind scheme (assuming velocity field = (1, 1, 1))
        float velocity = 1.0f;
        phi_new[idx] = phi[idx] - DT * velocity * (dphi_dx + dphi_dy + dphi_dz);
    }
}

int main() {
    size_t size = NX * NY * NZ * sizeof(float);

    // Host memory allocation
    float* h_phi = (float*)malloc(size);

    // Initial condition: Sphere level set
    for (int k = 0; k < NZ; k++) {
        for (int j = 0; j < NY; j++) {
            for (int i = 0; i < NX; i++) {
                int idx = (k * NY + j) * NX + i;
                float x = (i - NX / 2) * DX;
                float y = (j - NY / 2) * DX;
                float z = (k - NZ / 2) * DX;
                h_phi[idx] = sqrtf(x*x + y*y + z*z) - 0.2f; // Sphere of radius 0.2
            }
        }
    }

    // Device memory allocation
    float *d_phi, *d_phi_new;
    checkCuda(cudaMalloc((void**)&d_phi, size));
    checkCuda(cudaMalloc((void**)&d_phi_new, size));

    // Copy initial condition to device
    checkCuda(cudaMemcpy(d_phi, h_phi, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE, (NY + BLOCK_SIZE - 1) / BLOCK_SIZE, (NZ + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Time-stepping loop
    for (int step = 0; step < STEPS; step++) {
        level_set_kernel<<<numBlocks, threadsPerBlock>>>(d_phi, d_phi_new);
        checkCuda(cudaGetLastError());

        // Swap pointers
        float* temp = d_phi;
        d_phi = d_phi_new;
        d_phi_new = temp;
    }

    // Copy final result back to host
    checkCuda(cudaMemcpy(h_phi, d_phi, size, cudaMemcpyDeviceToHost));

    // Output a section of the level set field
    for (int k = NZ / 2 - 2; k <= NZ / 2 + 2; k++) {
        for (int j = NY / 2 - 2; j <= NY / 2 + 2; j++) {
            for (int i = NX / 2 - 2; i <= NX / 2 + 2; i++) {
                int idx = (k * NY + j) * NX + i;
                printf("%.3f ", h_phi[idx]);
            }
            printf("\n");
        }
        printf("\n");
    }

    // Free memory
    cudaFree(d_phi);
    cudaFree(d_phi_new);
    free(h_phi);

    return 0;
}
