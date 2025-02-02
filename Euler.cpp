#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define NX 256         // Grid size (X)
#define NY 256         // Grid size (Y)
#define STEPS 10000    // Number of time steps
#define BLOCK_SIZE 16  // CUDA block size
#define GAMMA 1.4f     // Ratio of specific heats
#define CFL 0.5f       // CFL condition number

// CUDA error checking macro
#define checkCuda(call) { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA Error: %s (error code: %d)\n", cudaGetErrorString(err), err); \
        exit(EXIT_FAILURE); \
    } \
}

// Compute flux using Roe approximate Riemann solver
__device__ float roe_flux(float rhoL, float uL, float vL, float EL, 
                          float rhoR, float uR, float vR, float ER, 
                          float nx, float ny) {
    float pL = (GAMMA - 1.0f) * (EL - 0.5f * rhoL * (uL * uL + vL * vL));
    float pR = (GAMMA - 1.0f) * (ER - 0.5f * rhoR * (uR * uR + vR * vR));

    float u_hat = 0.5f * (uL + uR);
    float v_hat = 0.5f * (vL + vR);
    float cL = sqrtf(GAMMA * pL / rhoL);
    float cR = sqrtf(GAMMA * pR / rhoR);
    float c_hat = 0.5f * (cL + cR);

    float flux = 0.5f * (rhoL * (uL * nx + vL * ny) + rhoR * (uR * nx + vR * ny))
                 - 0.5f * c_hat * (rhoR - rhoL);

    return flux;
}

// Adaptive time step based on CFL condition
__device__ float compute_dt(float dx, float dy, float u, float v, float c) {
    float max_speed = fabsf(u) + c;
    return CFL * fminf(dx, dy) / max_speed;
}

// CUDA kernel for Euler equations with Roe solver and adaptive time-stepping
__global__ void euler_solver(float* rho, float* u, float* v, float* E, float dx, float dy, float* dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * NX + i;

    if (i > 0 && i < NX - 1 && j > 0 && j < NY - 1) {
        float rho_ij = rho[idx];
        float u_ij = u[idx];
        float v_ij = v[idx];
        float E_ij = E[idx];

        float p = (GAMMA - 1.0f) * (E_ij - 0.5f * rho_ij * (u_ij * u_ij + v_ij * v_ij));
        float c = sqrtf(GAMMA * p / rho_ij);

        // Adaptive time step
        dt[idx] = compute_dt(dx, dy, u_ij, v_ij, c);

        int idx_left = j * NX + (i - 1);
        int idx_right = j * NX + (i + 1);
        int idx_down = (j - 1) * NX + i;
        int idx_up = (j + 1) * NX + i;

        // Roe fluxes in x and y directions
        float flux_x = roe_flux(rho[idx_left], u[idx_left], v[idx_left], E[idx_left],
                                rho[idx_right], u[idx_right], v[idx_right], E[idx_right], 1.0f, 0.0f);

        float flux_y = roe_flux(rho[idx_down], u[idx_down], v[idx_down], E[idx_down],
                                rho[idx_up], u[idx_up], v[idx_up], E[idx_up], 0.0f, 1.0f);

        rho[idx] -= (*dt) / dx * flux_x + (*dt) / dy * flux_y;
        u[idx]   -= (*dt) / dx * flux_x + (*dt) / dy * flux_y;
        v[idx]   -= (*dt) / dx * flux_x + (*dt) / dy * flux_y;
        E[idx]   -= (*dt) / dx * flux_x + (*dt) / dy * flux_y;
    }
}

int main() {
    const float dx = 0.01f, dy = 0.01f;
    float dt = 0.0001f;

    size_t size = NX * NY * sizeof(float);

    float *h_rho = (float*)malloc(size);
    float *h_u = (float*)malloc(size);
    float *h_v = (float*)malloc(size);
    float *h_E = (float*)malloc(size);

    for (int j = 0; j < NY; j++) {
        for (int i = 0; i < NX; i++) {
            int idx = j * NX + i;
            h_rho[idx] = 1.0f;
            h_u[idx] = (i == NX / 2 && j == NY / 2) ? 0.5f : 0.0f;
            h_v[idx] = 0.0f;
            h_E[idx] = 2.5f;
        }
    }

    float *d_rho, *d_u, *d_v, *d_E, *d_dt;
    checkCuda(cudaMalloc((void**)&d_rho, size));
    checkCuda(cudaMalloc((void**)&d_u, size));
    checkCuda(cudaMalloc((void**)&d_v, size));
    checkCuda(cudaMalloc((void**)&d_E, size));
    checkCuda(cudaMalloc((void**)&d_dt, size));

    checkCuda(cudaMemcpy(d_rho, h_rho, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_u, h_u, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_v, h_v, size, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_E, h_E, size, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((NX + BLOCK_SIZE - 1) / BLOCK_SIZE, (NY + BLOCK_SIZE - 1) / BLOCK_SIZE);

    for (int step = 0; step < STEPS; step++) {
        euler_solver<<<numBlocks, threadsPerBlock>>>(d_rho, d_u, d_v, d_E, dx, dy, d_dt);
        checkCuda(cudaGetLastError());
    }

    checkCuda(cudaMemcpy(h_rho, d_rho, size, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_u, d_u, size, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_v, d_v, size, cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(h_E, d_E, size, cudaMemcpyDeviceToHost));

    for (int j = NY / 2 - 5; j <= NY / 2 + 5; j++) {
        for (int i = NX / 2 - 5; i <= NX / 2 + 5; i++) {
            int idx = j * NX + i;
            printf("%.3f ", h_rho[idx]);
        }
        printf("\n");
    }

    cudaFree(d_rho);
    cudaFree(d_u);
    cudaFree(d_v);
    cudaFree(d_E);
    cudaFree(d_dt);
    free(h_rho);
    free(h_u);
    free(h_v);
    free(h_E);

    return 0;
}
