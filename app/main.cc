#include <stdio.h>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include "../include/ImageData.h"
#include "../include/weight_loader.h"
#include "../include/mnist.h"

//std::vector<float> log_softmax(const std::vector<float>& inputs) {
//    std::vector<float> result;
//    float max_val = *(std::max_element(inputs.begin(), inputs.end()));
//    float sum = 0.0;
//
//    for (auto input : inputs) {
//        float log_softmax_val = std::log(std::exp(input - max_val));
//        result.push_back(log_softmax_val);
//        sum += log_softmax_val;
//    }
//
//    for (float& val : result) {
//        val -= sum;
//    }
//
//    return result;
//}

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

void softmaxOnCPU(std::vector<float>& input, std::vector<float>& output,
                  int M, int N) {

    for (int i = 0; i < M; ++i) {
        float maxValue = input[i * M + 0];
        for (int j = 1; j < N; ++j) {
            if (maxValue < input[i * M + j])
                maxValue = input[i * M + j];
        }

        for (int j = 0; j < N; ++j) {
            output[i * N + j] = std::exp(input[i * N + j] - maxValue); //numer
        }
    }

    for (int i = 0; i < M; ++i) {
        float sum = 0.f; //denom
        for (int j = 0; j < N; ++j)
            sum += output[i * N + j];

        for (int j = 0; j < N; ++j)
            output[i * N + j] /= sum;
    }
}


int inferenceOnCPU(std::unique_ptr<Model>& model, ImageData& img, std::vector<int8_t>& labels) {
    const int num_bt = 1;
    const int inp_sz = img.cols * img.rows;
    std::vector<float> inp(num_bt * inp_sz);
    constexpr float MEAN = 0.1307f;
    constexpr float VAR = 0.3081f;

    //copy - transform
    for (int i = 0; i < inp.size(); ++i) {
        inp[i] = (img.data[i] - MEAN) / VAR;
    }

    auto& linears = model->linears;

    printf("======================================== \n");

    int M = num_bt;
    int K = inp_sz;
    int N = linears[0].col;
    assert(K == linears[0].row);

    std::vector<float> C0(M * N, 0.f);
    matMulOnCPU(inp, linears[0].w, C0, M, K, N);
    matAddBiasOnCPU(C0, linears[0].b, M, N);
    matReLUOnCPU(C0, M, N);
    printf("C0  relu %f ==\n", C0[0]);


    //lyr0 - lyr1
    K = N;
    N = linears[1].col;
    assert(K == linears[1].row);

    std::vector<float> C1(M * N, 0.f);
    matMulOnCPU(C0, linears[1].w, C1, M, K, N);
    matAddBiasOnCPU(C1, linears[1].b, M, N);
    matReLUOnCPU(C1, M, N);
    printf("C1  relu %f ==\n", C1[0], C1[1]);

    //lyr1 - lyr2
    K = N;
    N = linears[2].col;
    assert(K == linears[2].row);

    std::vector<float> C2(M * N, 0.f);
    matMulOnCPU(C1, linears[2].w, C2, M, K, N);
    matAddBiasOnCPU(C2, linears[2].b, M, N);
    //std::vector<float> softmax(M * N, 0.f);
    //softmaxOnCPU(C2, softmax, M, N);
    log_softmax1d(C2);

    for (int i = 0; i < 10; ++i) {
        printf("==== softmax[%d] : %f =====\n", i, C2[i]);
    }
//    [-25.5225, -21.9758, -22.0153, -19.5940, -27.5558, -25.8997, -32.2494,
//            0.0000, -26.3183, -18.2736]

//
//    int ret = 0;
//    for (int i = 1; i < 10; ++i) {
//        if (softmax[ret] < softmax[i]) {
//            ret = i;
//        }
//    }
//    return ret;
}



int main() {

//    load_mnist();
//
//    // print pixels of first data in test dataset
//    for (int i = 0; i < 784; i++) {
//        printf("%1.1f ", test_image[0][i]);
//        if ((i+1) % 28 == 0) putchar('\n');
//    }


    std::string imageFile = "/tmp/tmp.3nS42pXqhM/res/t10k-images-idx3-ubyte";
    std::string labelFile = "/tmp/tmp.3nS42pXqhM/res/t10k-labels-idx1-ubyte";
    std::string weightFile = "/tmp/tmp.3nS42pXqhM/res/torch_weights.json";

    ImageData imgData(imageFile);
    std::vector<int8_t> labels = loadLabel(labelFile);

    WeightLoader wl;
    auto model = wl.load(weightFile);


    inferenceOnCPU(model, imgData, labels);

    //printf("images %d %d \n", imgData.rows * imgData.cols * imgData.count,   imgData.data.size());
//    const auto& w = model->linears.back().w;
//    printf("==== wgt %d %d ==== \n", w.size(), model->linears[2].col * model->linears[2].row);

    return 0;
}
