#include "gpu_inference.h"
#include "ImageData.h"
#include "weight_loader.h"
#include <stdio.h>

constexpr int IMG_SZ = 784;
#define BLK_SZ 16


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


    //mem X
    float* X;
    memcpyGPU(&X, img.data.data(), sizeof(float) * numBt * inpSz);




    int M = numBt;
    int K = inpSz;
    int N = linears[0].col;

    float *W, *B, *out;
    memcpyGPU(&W, linears[0].w.data(), sizeof(float) * K * N);
    memcpyGPU(&B, linears[0].b.data(), sizeof(float) * K);
    memsetGPU(&out, sizeof(float) * M * N);

//    dim3 dimGrid(1, 1, 1);
//    dim3 dimBlock(M, N);
    dim3 gridDim(ceil((float)M / BLK_SZ), ceil((float)N / BLK_SZ));
    dim3 blockDim(BLK_SZ, BLK_SZ);
    linearWithReLU <<<gridDim,  blockDim>>> (X, W, B, out, M, K, N);
//    test <<<1, 10>>>(out);

    cudaDeviceSynchronize();

    float* outHost = new float[M * N];
    cudaMemcpy(outHost, out, sizeof(float) *  M * N, cudaMemcpyDeviceToHost);

    printf("CS=== %f %f %f ====\n", outHost[0], outHost[1], outHost[2]);

}
