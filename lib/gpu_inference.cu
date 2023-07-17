#include "gpu_inference.h"
#include "ImageData.h"
#include "weight_loader.h"
#include <stdio.h>
#include <array>
#include <cmath>
#include <cassert>

constexpr int IMG_SZ = 784;
#define BLK_SZ 16


void log_softmax1d(float* input, int sz) {
    float sum = 0.0f;

    for (int i = 0; i < sz; i++) {
        input[i] = std::exp(input[i]);
        sum += input[i];
    }

    for (int i = 0; i < sz; i++) {
        input[i] /= sum;
        input[i] = log(input[i]);
    }
}

__global__ void test(float* out)
{
    out[threadIdx.x] = 10.0;
    printf("kkkkk12\n");
}

__global__ void linearWithReLU(float* X, float* W, float* B, float* out,
                                int M, int K, int N) {


//    int c = BLK_SZ * blockIdx.x + threadIdx.x;
//    int r = BLK_SZ * blockIdx.y + threadIdx.y;
//    if (r >= M || c >= N)
//        return;
//    int c_idx = r * N + c;
//    for (int i = 0; i < K; ++i) {
//        out[c_idx] += X[r * K + i] * W[i * N + c];
//    }

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float sX[BLK_SZ][BLK_SZ];
    __shared__ float sW[BLK_SZ][BLK_SZ];

    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

    float acc = 0.f;

    for (int b = 0; b < ceil( (float)K / BLK_SZ ); ++b) {
        int offs = b * BLK_SZ;

        if (row >= M || offs + localCol >= K)
            sX[localRow][localCol] = 0.f;
        else
            sX[localRow][localCol] = X[row * K + (offs + localCol)];

        if (col >= N || offs + localRow >= K)
            sW[localRow][localCol] = 0.f;
        else
            sW[localRow][localCol] = W[(offs + localRow) * N + col];


        __syncthreads();

        for (int k = 0; k < BLK_SZ; ++k) {
            acc += sX[localRow][k] * sW[k][localCol];
        }

        __syncthreads();
    }

    if (row >= M || col >= N)
        return;

    float ret = acc + B[col];
    if (ret < 0.f)
        ret = 0.f;

    out[row * N + col] = ret;
}


__global__ void linear(float* X, float* W, float* B, float* out,
                       int M, int K, int N) {

    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ float sX[BLK_SZ][BLK_SZ];
    __shared__ float sW[BLK_SZ][BLK_SZ];

    int localRow = threadIdx.y;
    int localCol = threadIdx.x;

    float acc = 0.f;

    for (int b = 0; b < ceil( (float)K / BLK_SZ ); ++b) {
        int offs = b * BLK_SZ;

        if (row >= M || offs + localCol >= K)
            sX[localRow][localCol] = 0.f;
        else
            sX[localRow][localCol] = X[row * K + (offs + localCol)];

        if (col >= N || offs + localRow >= K)
            sW[localRow][localCol] = 0.f;
        else
            sW[localRow][localCol] = W[(offs + localRow) * N + col];


        __syncthreads();

        for (int k = 0; k < BLK_SZ; ++k) {
            acc += sX[localRow][k] * sW[k][localCol];
        }

        __syncthreads();
    }

    if (row >= M || col >= N)
        return;

    float ret = acc + B[col];

    out[row * N + col] = ret;
}


void inferenceOnGPU(std::unique_ptr<Model>& model, ImageData& img, int imgIdx, int numBt) {

    auto memcpyGPU = [](float** p, float* data, int memSz) {
        cudaMalloc(p, memSz);
        cudaMemset(*p, 0.0, memSz);
        cudaMemcpy(*p, data, memSz, cudaMemcpyHostToDevice);
    };

    auto memsetGPU = [](float** p, int memSz) {
        cudaMalloc(p, memSz);
        cudaMemset(*p, 0, memSz);
    };

    auto& linears = model->linears;
    const int inpSz = img.cols * img.rows;


    struct MKN {int m; int k; int n;};
    std::array<MKN, 3> mknArr = {
            MKN{numBt, inpSz, linears[0].col},
            MKN{numBt, linears[0].col, linears[1].col},
            MKN{numBt, linears[1].col, linears[2].col}
    };

    float* X;
    memcpyGPU(&X, img.data.data(), sizeof(float) * numBt * inpSz);

    for (int i = 0; i < linears.size(); ++i) {
        int M = mknArr[i].m, K = mknArr[i].k, N = mknArr[i].n;

        float *W, *B, *out;
        memcpyGPU(&W, linears[i].w.data(), sizeof(float) * K * N);
        memcpyGPU(&B, linears[i].b.data(), sizeof(float) * K);
        memsetGPU(&out, sizeof(float) * M * N);

        dim3 gridDim(ceil((float)M / BLK_SZ), ceil((float)N / BLK_SZ));
        dim3 blockDim(BLK_SZ, BLK_SZ);

        if (i < linears.size() - 1) {
            linearWithReLU <<<gridDim,  blockDim>>> (X, W, B, out, M, K, N);
        } else {
            linear <<<gridDim,  blockDim>>> (X, W, B, out, M, K, N);
        }

        X = out;
    }

    cudaDeviceSynchronize();

    float* outHost = new float[numBt * 10];
    cudaMemcpy(outHost, X, sizeof(float) * numBt * 10, cudaMemcpyDeviceToHost);
    //softmax
    log_softmax1d(outHost, 10);

    //test
    float firstRowSoftMax[10] = {
            -25.5225, -21.9758, -22.0153, -19.5940, -27.5558, -25.8997, -32.2494,
            0.0000, -26.3183, -18.2736
    };

    for (int i = 0; i < 10; ++i) {
        assert(abs(outHost[i] - firstRowSoftMax[i]) > 0.0001);
    }

    printf("========Pass!!!======\n");


//    int M = numBt;
//    int K = inpSz;
//    int N = linears[0].col;
//
//    float *W, *B, *out;
//    memcpyGPU(&W, linears[0].w.data(), sizeof(float) * K * N);
//    memcpyGPU(&B, linears[0].b.data(), sizeof(float) * K);
//    memsetGPU(&out, sizeof(float) * M * N);
//
//    dim3 gridDim(ceil((float)M / BLK_SZ), ceil((float)N / BLK_SZ));
//    dim3 blockDim(BLK_SZ, BLK_SZ);
//    linearWithReLU <<<gridDim,  blockDim>>> (X, W, B, out, M, K, N);
//
//    cudaDeviceSynchronize();
//
//    float* outHost = new float[M * N];
//    cudaMemcpy(outHost, out, sizeof(float) *  M * N, cudaMemcpyDeviceToHost);
//
//    printf("CS=== %f %f %f ====\n", outHost[0], outHost[1], outHost[2]);

}
