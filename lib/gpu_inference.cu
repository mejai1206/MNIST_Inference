#include "gpu_inference.cuh"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "ImageData.h"
#include "weight_loader.h"
#include <stdio.h>
#include <array>
#include <cmath>
#include <cassert>

#define CDIV(X, Y) (((X) + (Y) - 1) / (Y))

#define BLK_SZ 16

cudaEvent_t finishEvt = NULL;

__global__ void linearKernel(float* X, float* W, float* B, float* out,
                              int M, int K, int N, bool relu) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;

    __shared__ float sX[BLK_SZ][BLK_SZ];
    __shared__ float sW[BLK_SZ][BLK_SZ];

    int localCol = threadIdx.x;
    int localRow = threadIdx.y;

    float acc = 0.f;

    int bSize = CDIV(K, BLK_SZ);

    for (int b = 0; b < bSize; ++b) {
        int offs = b * BLK_SZ;

        if (row >= M || offs + localCol >= K) {
            sX[localRow][localCol] = 0.f;
        } else {
            sX[localRow][localCol] = X[row * K + (offs + localCol)];
        }

        if (col >= N || offs + localRow >= K) {
            sW[localRow][localCol] = 0.f;
        } else {
            sW[localRow][localCol] = W[(offs + localRow) * N + col];
        }

        __syncthreads();

        for (int k = 0; k < BLK_SZ; ++k) {
            acc += sX[localRow][k] * sW[k][localCol];
        }

        __syncthreads();
    }

    if (row >= M || col >= N)
        return;

    float ret = acc + B[col];

    if (relu && ret < 0.f) {
        ret = 0.f;
    }

    out[row * N + col] = ret;
}


__global__ void matchCountKernel(float* X, int8_t* labels, int* count, int M, int N, int imgIdx) {
    int row = threadIdx.x;
    int retIdx = row * N;
    int retLabel = 0;

    for (int i = 1; i < N; ++i) { //10 shared
        int idx = row * N + i;
        if (X[retIdx] < X[idx]) {
            retIdx = idx;
            retLabel = i;
        }
    }

    if (retLabel == labels[imgIdx + row]) {
        atomicAdd(count, 1); //reduction
    }
}


template <typename T>
void memcpyGPU(T** p, const T* data, int memSz) {
    cudaMalloc(p, memSz);
    cudaMemset(*p, 0, memSz);
    cudaMemcpy(*p, data, memSz, cudaMemcpyHostToDevice);
    ASSERT_CUDA;
}

template <typename T>
void memsetGPU(T** p, int memSz) {
    cudaMalloc(p, memSz);
    cudaMemset(*p, 0, memSz);
    ASSERT_CUDA;
};

InferenceManager::InferenceManager(Model* model, int inpSz, int numBt,
                                   const std::vector<int8_t>& labels) : m_model(model), m_inpSz(inpSz), m_numBt(numBt) {
    auto& linears = model->linears;

    m_mkn.push_back({numBt, inpSz, linears[0].col});

    for (int i = 1; i < linears.size(); ++i) {
        m_mkn.push_back({numBt, linears[i-1].col, linears[i].col});
    }

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
    memcpyGPU(&m_labelBuffer, labels.data(), sizeof(int8_t) * labels.size());
    memsetGPU(&m_pCnt, sizeof(int));
}


void InferenceManager::inferenceOnGPU(ImageData& img, int imgIdx, std::vector<int8_t>& labels) {

    auto& linears = m_model->linears;

    cudaMemcpy(m_inpBuffer, img.data.data() + (imgIdx * m_inpSz),
               sizeof(float) * m_numBt * m_inpSz, cudaMemcpyHostToDevice);
    ASSERT_CUDA;

    for (int i = 0; i < linears.size(); ++i) {
        int M = m_mkn[i].m;
        int K = m_mkn[i].k;
        int N = m_mkn[i].n;

        float* X = (i == 0) ? m_inpBuffer : m_outBuffers[i-1];
        float* W = m_wBuffers[i];
        float* B = m_bBuffers[i];
        float* out = m_outBuffers[i];

        dim3 gridDim(CDIV(N, BLK_SZ), CDIV(M, BLK_SZ)); //y-row, x-col
        dim3 blockDim(BLK_SZ, BLK_SZ);
        linearKernel <<<gridDim,  blockDim>>>(X, W, B, out, M, K, N, i < linears.size() - 1);
        ASSERT_CUDA;
    }

    dim3 gridDim(1);
    dim3 blockDim(m_numBt);
    matchCountKernel <<<gridDim, blockDim>>>(m_outBuffers.back(), m_labelBuffer, m_pCnt, m_numBt, 10, imgIdx);

    bool isLastOperation = (imgIdx == 10000 - m_numBt);
    if (isLastOperation) {
        cudaEventCreate(&finishEvt);
        cudaEventRecord(finishEvt);
    }
}

bool InferenceManager::checkFinish() {
    assert(finishEvt != NULL);
    bool isFinish = cudaEventQuery(finishEvt) == cudaSuccess;

    if (isFinish) {
        int cntHost = 0;
        cudaMemcpy(&cntHost, m_pCnt, sizeof(int), cudaMemcpyDeviceToHost);
        m_matchCount = cntHost;
    }

    return isFinish;
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

    if (m_labelBuffer) {
        cudaFree(m_labelBuffer);
    }

    if (m_labelBuffer) {
        cudaFree(m_labelBuffer);
    }

    if (m_pCnt) {
        cudaFree(m_pCnt);
    }

    cudaEventDestroy(finishEvt);
    finishEvt = NULL;
}