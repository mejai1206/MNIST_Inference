#include "gpu_inference.h"
#include "ImageData.h"
#include "weight_loader.h"
#include <stdio.h>
#include <array>
#include <cmath>
#include <cassert>

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

//__global__ void linearWithReLU(float* X, float* W, float* B, float* out,
//                                int M, int K, int N) {
//    int row = blockDim.y * blockIdx.y + threadIdx.y;
//    int col = blockDim.x * blockIdx.x + threadIdx.x;
//
//    __shared__ float sX[BLK_SZ][BLK_SZ];
//    __shared__ float sW[BLK_SZ][BLK_SZ];
//
//    int localRow = threadIdx.y;
//    int localCol = threadIdx.x;
//
//    float acc = 0.f;
//
//    for (int b = 0; b < ceil( (float)K / BLK_SZ ); ++b) {
//        int offs = b * BLK_SZ;
//
//        if (row >= M || offs + localCol >= K)
//            sX[localRow][localCol] = 0.f;
//        else
//            sX[localRow][localCol] = X[row * K + (offs + localCol)];
//
//        if (col >= N || offs + localRow >= K)
//            sW[localRow][localCol] = 0.f;
//        else
//            sW[localRow][localCol] = W[(offs + localRow) * N + col];
//
//
//        __syncthreads();
//
//        for (int k = 0; k < BLK_SZ; ++k) {
//            acc += sX[localRow][k] * sW[k][localCol];
//        }
//
//        __syncthreads();
//    }
//
//    if (row >= M || col >= N)
//        return;
//
//    float ret = acc + B[col];
//    if (ret < 0.f)
//        ret = 0.f;
//
//    out[row * N + col] = ret;
//}


__global__ void linear(float* X, float* W, float* B, float* out,
                       int M, int K, int N) {

    int r = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    if (r >= M || c >= N)
        return;


    double acc = 0.0;
    for (int k = 0; k < K; ++k) {
        acc += (X[r * K + k] * W[k * N + c]);
    }

    acc += B[c];
    out[r * N + c] = acc;
}


__global__ void linearReLU(float* X, float* W, float* B, float* out,
                       int M, int K, int N) {

    int r = blockDim.x * blockIdx.x + threadIdx.x;
    int c = blockDim.y * blockIdx.y + threadIdx.y;

    if (r >= M || c >= N)
        return;


    float acc = 0.f;
    for (int k = 0; k < K; ++k) {
        acc += (X[r * K + k] * W[k * N + c]);
    }

    acc += B[c];

    if (acc < 0.f)
        acc = 0.f;

    out[r * N + c] = acc;
}

InferenceManager::InferenceManager(Model* model, int imgCnt, int inpSz, int numBt) : m_model(model), m_imgCnt(imgCnt), m_inpSz(inpSz), m_numBt(numBt) {

    auto& linears = model->linears;

    m_mkn.push_back({numBt, inpSz, linears[0].col});

    for (int i = 1; i < linears.size(); ++i) {
        m_mkn.push_back({numBt, linears[i-1].col, linears[i].col});
    }

    auto memcpyGPU = [](float** p, float* data, int memSz) {
        cudaMalloc(p, memSz);
        cudaMemset(*p, 0, memSz);
        cudaMemcpy(*p, data, memSz, cudaMemcpyHostToDevice);
        assert(cudaGetLastError() == cudaSuccess);
    };

    auto memsetGPU = [](float** p, int memSz) {
        cudaMalloc(p, memSz);
        cudaMemset(*p, 0, memSz);
        assert(cudaGetLastError() == cudaSuccess);
    };

    for (int i = 0; i < linears.size(); ++i) {
        int M = m_mkn[i].m;
        int K = m_mkn[i].k;
        int N = m_mkn[i].n;

        float *W, *B, *out;
        memcpyGPU(&W, linears[i].w.data(), sizeof(float) * N * K);
        memcpyGPU(&B, linears[i].b.data(), sizeof(float) * K);
        memsetGPU(&out, sizeof(float) * M * N);

        m_wBuffers.push_back(W);
        m_bBuffers.push_back(B);
        m_outBuffers.push_back(out);
    }

    memsetGPU(&m_inpBuffer, sizeof(float) * m_numBt * m_inpSz);
}


void InferenceManager::inferenceOnGPU(ImageData& img, int imgIdx, std::vector<int8_t>& labels) {

    auto& linears = m_model->linears;

    cudaMemcpy(m_inpBuffer, img.data.data() + (imgIdx * m_numBt * m_inpSz),
               sizeof(float) * m_numBt * m_inpSz, cudaMemcpyHostToDevice);

    for (int i = 0; i < linears.size(); ++i) {
        int M = m_mkn[i].m;
        int K = m_mkn[i].k;
        int N = m_mkn[i].n;

        dim3 gridDim(ceil((float)M / BLK_SZ), ceil((float)N / BLK_SZ));
        dim3 blockDim(BLK_SZ, BLK_SZ);

//        printf("[%d]  gdim.x: %d  gdim.y: %d =====\n", i, gridDim.x, gridDim.y); //(1, 64)

        float* X = (i == 0) ? m_inpBuffer : m_outBuffers[i-1];
        float* W = m_wBuffers[i];
        float* B = m_bBuffers[i];
        float* out = m_outBuffers[i];

        if (i < linears.size() - 1) {
            linearReLU <<<gridDim,  blockDim>>> (X, W, B, out, M, K, N);
        } else {
            linear <<<gridDim,  blockDim>>> (X, W, B, out, M, K, N);
        }

        cudaDeviceSynchronize();
        assert(cudaGetLastError() == cudaSuccess);

//        bool test = true;
//        if (test) {
//            if (i == 1) {
//                float* tmp_out = new float[sizeof(float) * M * N];
//                cudaMemcpy(tmp_out, out, sizeof(float) * M * N, cudaMemcpyDeviceToHost);
//                printf("GPU==== C1 [%d] %f %f %f \n\n", i, tmp_out[0], tmp_out[1], tmp_out[2]);
//                delete[] tmp_out;
//            }
//        }
    }

    cudaDeviceSynchronize();

    float* outHost = new float[m_numBt * 10];
    cudaMemcpy(outHost, m_outBuffers.back(), sizeof(float) * m_numBt * 10, cudaMemcpyDeviceToHost);
    log_softmax1d(outHost, 10);


    ///test////
//    float firstRowSoftMax[10] = {
//            -25.5225, -21.9758, -22.0153, -19.5940, -27.5558, -25.8997, -32.2494,
//            0.0000, -26.3183, -18.2736
//    };

//    for (int i = 0; i < 10; ++i) {
//        printf("====== [%d]: %f ====\n", i, outHost[i]);
//        assert(abs(outHost[i] - firstRowSoftMax[i]) <= 0.01);
//    }
//    printf("========Pass77!!!======\n");
    /////////




    int ret = 0;
    for (int i = 1; i < 10; ++i) {
        if (outHost[ret] < outHost[i]) {
            ret = i;
        }
    }

    if (ret == labels[imgIdx])
        m_matchCount++;
    else
        m_noMat++;


    delete[] outHost;
}

InferenceManager::~InferenceManager() {
    for (int i = 0; i < m_wBuffers.size(); ++i) {
        cudaFree(m_wBuffers[i]);
        cudaFree(m_bBuffers[i]);
        cudaFree(m_outBuffers[i]);
    }

    if (m_inpBuffer) {
        cudaFree(m_inpBuffer);
    }
}
