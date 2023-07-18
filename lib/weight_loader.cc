#include "weight_loader.h"

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <nlohmann/json.hpp>

using json = nlohmann::json;

std::unique_ptr<Model> WeightLoader::load(std::string filePath) {
    std::ifstream ifs(filePath);
    std::stringstream buffer;
    buffer << ifs.rdbuf();
    std::string buf = buffer.str();

    auto j = json::parse(buf);

    constexpr int LINEAR_COUNT = 3;
    const char* weightKeys[LINEAR_COUNT] = {"fc1.weight", "fc2.weight", "fc3.weight"};
    const char* biasKeys[LINEAR_COUNT] = {"fc1.bias", "fc2.bias", "fc3.bias"};

    auto model = std::make_unique<Model>();

    // trans
    for (int i = 0; i < LINEAR_COUNT; ++i) {
        const auto& w = j[weightKeys[i]];
        const auto& b = j[biasKeys[i]];

        Linear linear;

        int NUM_ROW = w.size();
        int NUM_COL = w[0].size();

        linear.row = NUM_COL;
        linear.col = NUM_ROW;

        for (int col = 0; col < NUM_COL; ++col) {
            for (int row = 0; row < NUM_ROW; ++row) {
                linear.w.push_back(w[row][col]);
            }
        }

        for (int i = 0; i < b.size(); ++i) {
            linear.b.push_back(b[i]);
        }


        model->linears.emplace_back(linear);
    }

    return model;
}


