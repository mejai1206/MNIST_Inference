#ifndef CMAKE_FOR_ALL_WEIGHTLOADER_H
#define CMAKE_FOR_ALL_WEIGHTLOADER_H
#include <string>
#include <vector>
#include <memory>

class Linear {
public:
    int row;
    int col;
    std::vector<float> w;
    std::vector<float> b;
};

class Model {
public:
    std::vector<Linear> linears;
};


class WeightLoader {
public:
    static std::unique_ptr<Model> load(std::string filePath);
};


#endif //CMAKE_FOR_ALL_WEIGHTLOADER_H
