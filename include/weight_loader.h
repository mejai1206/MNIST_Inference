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

    void printFirstRow() {
        for (int i = 0; i < 5; ++i)
            printf("w_row[%d] : %f \n", i, w[i]);
    }

    void printFirstCol() {
        for (int i = 0; i < 5; ++i)
            printf("w_col[%d] : %f \n", i, w[i * col]);
    }

};

class Model {
public:
    std::vector<Linear> linears;

    void printShape() {
        for (int i = 0; i < linears.size(); ++i) {
            printf("row: %d  col: %d \n", linears[i].row, linears[i].col);
        }
    }
};


class WeightLoader {
public:
    std::unique_ptr<Model> load(std::string filePath);
};


#endif //CMAKE_FOR_ALL_WEIGHTLOADER_H
