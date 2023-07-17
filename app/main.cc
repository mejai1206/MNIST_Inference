#include <stdio.h>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include "../include/ImageData.h"
#include "../include/weight_loader.h"
#include "../include/gpu_inference.h"


void log_softmax1d(std::vector<float>& input) {
    float sum = 0.0f;

    for (int i = 0; i < input.size(); i++) {
        input[i] = std::exp(input[i]);
        sum += input[i];
    }

    for (int i = 0; i < input.size(); i++) {
        input[i] /= sum;
        input[i] = log(input[i]);
    }
}

void matMulOnCPU(std::vector<float>& A,
                 std::vector<float>& B,
                 std::vector<float>& C,
                 int M, int K, int N) {

    for (int row = 0; row < M; row++) {
        for (int col = 0; col < N; col++) {
            C[row * N + col] = 0.f;
            for (int k = 0; k < K; ++k) {
                C[row * N + col] += (A[row * K + k] * B[k * N + col]); //C(M,N)
            }
        }
    }
}


void matAddBiasOnCPU(std::vector<float>& C, std::vector<float>& bias,
                     int M, int N) {

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; col++) {
            C[row * N + col] += bias[col];
        }
    }
}

void matReLUOnCPU(std::vector<float>& C, int M, int N) {
    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; col++) {
            int cIndex = row * N + col;
            if (C[cIndex] < 0.f)
                C[cIndex] = 0.f;
        }
    }
}

std::pair<std::vector<float>, int> inferenceOnCPU(std::unique_ptr<Model>& model, ImageData& img, int imgIdx) {
    const int num_bt = 1;
    const int inp_sz = img.cols * img.rows; //784
    std::vector<float> inp(num_bt * inp_sz);

    //copy - transform
    for (int i = 0; i < inp.size(); ++i) {
        inp[i] = img.data[(imgIdx * inp_sz) + i];
    }

    auto& linears = model->linears;

    //printf("======================================== \n");

    int M = num_bt;
    int K = inp_sz;
    int N = linears[0].col;
    assert(K == linears[0].row);

    std::vector<float> C0(M * N, 0.f);
    matMulOnCPU(inp, linears[0].w, C0, M, K, N);
    matAddBiasOnCPU(C0, linears[0].b, M, N);
    //printf("===== C0: %f %f %f ==\n", C0[0], C0[1], C0[2]);
    matReLUOnCPU(C0, M, N);
    printf("===== C0: %f %f %f ==\n", C0[0], C0[1], C0[2]);

    //lyr0 - lyr1
    K = N;
    N = linears[1].col;
    assert(K == linears[1].row);

    std::vector<float> C1(M * N, 0.f);
    matMulOnCPU(C0, linears[1].w, C1, M, K, N);
    matAddBiasOnCPU(C1, linears[1].b, M, N);
    matReLUOnCPU(C1, M, N);

    //lyr1 - lyr2
    K = N;
    N = linears[2].col;
    assert(K == linears[2].row);

    std::vector<float> C2(M * N, 0.f);
    matMulOnCPU(C1, linears[2].w, C2, M, K, N);
    matAddBiasOnCPU(C2, linears[2].b, M, N);
    log_softmax1d(C2);

    int ret = 0;
    for (int i = 1; i < 10; ++i) {
        if (C2[ret] < C2[i]) {
            ret = i;
        }
    }

    return {C2, ret};
}


bool testCPU(std::unique_ptr<Model>& model, ImageData& img) {
    auto softMaxRet = inferenceOnCPU(model, img, 0);

    float firstRowSoftMax[10] = {
            -25.5225, -21.9758, -22.0153, -19.5940, -27.5558, -25.8997, -32.2494,
            0.0000, -26.3183, -18.2736
    };

    for (int i = 0; i < 10; ++i)
        if (abs(softMaxRet.first[i] - firstRowSoftMax[i]) > 0.0001)
            return false;

    return true;
}

/// utils
std::vector<int8_t> loadLabel(std::string file) {
    auto fd = open(file.c_str(), O_RDONLY);
    assert(fd > 0);

    int32_t tmp = 0;
    read(fd, &tmp, sizeof(tmp));
    int32_t magic = be32toh(tmp);

    read(fd, &tmp, sizeof(tmp));
    int32_t count = be32toh(tmp);

    std::vector<int8_t> labels;
    for(int i=0; i < count; ++i) {
        int8_t label;
        read(fd, &label, sizeof(label));
        labels.emplace_back(label);
    }
    printf("magic: %x, count: %d\n", magic, count);
    close(fd);
    return labels;
}



int main() {

    std::string imageFile = "/tmp/tmp.3nS42pXqhM/res/t10k-images-idx3-ubyte";
    std::string labelFile = "/tmp/tmp.3nS42pXqhM/res/t10k-labels-idx1-ubyte";
    std::string weightFile = "/tmp/tmp.3nS42pXqhM/res/torch_weights.json";

    ImageData imgData(imageFile);
    std::vector<int8_t> labels = loadLabel(labelFile);

    WeightLoader wl;
    auto model = wl.load(weightFile);

    //assert(testCPU(model, imgData));
    testCPU(model, imgData);





//    auto start = std::chrono::system_clock::now();
//    int match = 0;
//    int imageCount = imgData.count;
//    for (int i = 0; i < imageCount; ++i) {
//        int ret = inferenceOnCPU(model, imgData, i).second;
//        if (ret == labels[i])
//            match++;
//        printf("===== %d ====\n", i);
//    }
//
//    auto end = std::chrono::system_clock::now();
//    auto micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
//    auto accuracy = match * 100.f / static_cast<float>(imageCount);
//    std::cout << "elapsed : " << micro.count() << "us" << std::endl;
//    std::cout << "accuracy: " << accuracy << "%" << std::endl;












//inf gpu


    int imageCount = imgData.count;

    auto start = std::chrono::system_clock::now();

    int batchSize = 100;

    InferenceManager im(model.get(), 784, batchSize);

    for (int i = 0; i < imageCount; i += batchSize) {
        im.inferenceOnGPU(imgData, i, labels);
        printf("=====dd %d ====\n", i);
    }

    auto end = std::chrono::system_clock::now();
    auto micro = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    auto accuracy = im.matchCount() * 100.f / static_cast<float>(imageCount);
    std::cout << "elapsed : " << micro.count() << "us" << std::endl;
    std::cout << "accuracy: " << accuracy << "%" << std::endl;

    printf("mat: %d   nomat: %d  =====\n", im.matchCount(), im.noMat());



    return 0;
}
