#include <stdio.h>
#include <iostream>
#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <chrono>
#include "../include/image_data.h"
#include "../include/weight_loader.h"
#include "../include/gpu_inference.cuh"

const std::string imageFile = "/tmp/tmp.3nS42pXqhM/res/t10k-images-idx3-ubyte";
const std::string labelFile = "/tmp/tmp.3nS42pXqhM/res/t10k-labels-idx1-ubyte";
const std::string weightFile = "/tmp/tmp.3nS42pXqhM/res/torch_weights.json";

void loadLabel(std::string file, std::vector<int8_t>& labels) {
    auto fd = open(file.c_str(), O_RDONLY);
    assert(fd > 0);

    int32_t tmp = 0;
    read(fd, &tmp, sizeof(tmp));
    int32_t magic = be32toh(tmp);

    read(fd, &tmp, sizeof(tmp));
    int32_t count = be32toh(tmp);

    for(int i=0; i < count; ++i) {
        int8_t label;
        read(fd, &label, sizeof(label));
        labels.push_back(label);
    }

    close(fd);
}

class Timer {
public:
    Timer() {
        m_start = std::chrono::system_clock::now();
    }

    void printElapsed(const char* tag) {
        auto end = std::chrono::system_clock::now();
        auto du = std::chrono::duration_cast<std::chrono::milliseconds>(end - m_start);
        std::cout << tag << du.count() << "ms" << std::endl;
    }

    std::chrono::time_point<std::chrono::system_clock> m_start;
};


int main() {
    Timer resTimer;

    ImageData imgData(imageFile);

    std::vector<int8_t> labels;
    loadLabel(labelFile, labels);

    auto model = WeightLoader::load(weightFile);

    resTimer.printElapsed("Resource File reading time: ");

    Timer inferenceMeomoryLoadTimer;

    int imageCount = imgData.count;
    int batchSize = 1;

    InferenceManager inferenceManager(model.get(), imgData.rows * imgData.cols, batchSize, labels);
    inferenceMeomoryLoadTimer.printElapsed("inference memory loading time: ");

    Timer inferenceTimer;
    for (int i = 0; i < imageCount; i += batchSize) {
        inferenceManager.inferenceOnGPU(imgData, i, labels);
    }

    inferenceTimer.printElapsed("inference start ~ end(host): ");

    while(!inferenceManager.checkFinish()) { }

    inferenceTimer.printElapsed("inference start ~ end(device waiting): ");


    auto accuracy = inferenceManager.matchCount() * 100.f / static_cast<float>(imageCount);
    std::cout << "accuracy: " << accuracy << "%" << std::endl;

    return 0;
}
