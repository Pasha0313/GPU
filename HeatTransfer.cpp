#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define N 256         // Grid size (NxN)
#define STEPS 1000    // Number of time steps
#define BLOCK_SIZE 16 // CUDA block size

// CUDA error checking macro
#define checkCuda(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code: %d)\n", cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    } \
}

// CUDA kernel for heat transfer with Dirichlet boundary conditions
__global__ void heat_transfer(float* T_new, float* T_old, float alpha, float dx, float dy, float dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    int idx = j * N + i;

    if (i > 0 && i < N-1 && j > 0 && j < N-1) {
        // 2D Finite Difference Scheme
        float d2T_dx2 = (T_old[idx + 1] - 2.0f * T_old[idx] + T_old[idx - 1]) / (dx * dx);
        float d2T_dy2 = (T_old[idx + N] - 2.0f * T_old[idx] + T_old[idx - N]) / (dy * dy);

        // Update temperature
        T_new[idx] = T_old[idx] + alpha * dt * (d2T_dx2 + d2T_dy2);
    } else {
        // Dirichlet boundary conditions (fixed temperature at boundaries)
        T_new[idx] = 0.0f;
    }
}

int main() {
    const float alpha = 0.01f;    // Thermal diffusivity
    const float dx = 0.01f, dy = 0.01f;
    const float dt = 0.0001f;     // Time step

    size_t size = N * N * sizeof(float);

    // Host memory allocation
    float *h_T = (float*)malloc(size);

    // Initial condition: Set temperature to 0
    for (int i = 0; i < N * N; i++) {
        h_T[i] = 0.0f;
    }
    
    // Hot spot in the center
    h_T[(N/2) * N + (N/2)] = 100.0f;

    // Device memory allocation
    float *d_T_old, *d_T_new;
    checkCuda(cudaMalloc((void**)&d_T_old, size));
    checkCuda(cudaMalloc((void**)&d_T_new, size));

    // Copy initial condition to device
    checkCuda(cudaMemcpy(d_T_old, h_T, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Time-stepping loop
    for (int step = 0; step < STEPS; step++) {
        heat_transfer<<<numBlocks, threadsPerBlock>>>(d_T_new, d_T_old, alpha, dx, dy, dt);
        checkCuda(cudaGetLastError());

        // Swap pointers
        float* temp = d_T_old;
        d_T_old = d_T_new;
        d_T_new = temp;
    }

    // Copy final result back to host
    checkCuda(cudaMemcpy(h_T, d_T_old, size, cudaMemcpyDeviceToHost));

    // Output a section of the temperature field
    for (int j = N/2 - 5; j <= N/2 + 5; j++) {
        for (int i = N/2 - 5; i <= N/2 + 5; i++) {
            printf("%6.2f ", h_T[j * N + i]);
        }
        printf("\n");
    }

    // Free memory
    cudaFree(d_T_old);
    cudaFree(d_T_new);
    free(h_T);

    return 0;
}
