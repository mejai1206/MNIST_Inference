#ifndef NUMBERCLASSIFIER_IMAGEDATA_H
#define NUMBERCLASSIFIER_IMAGEDATA_H

#include <fstream>
#include <sstream>
//#include <rapidjson/document.h>
#include <vector>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cmath>
#include <cassert>

class ImageData {
public:
    ImageData() {}

    ImageData(std::string file) {
        auto fd = open(file.c_str(), O_RDONLY);
        assert(fd > 0);

        int32_t tmp = 0;
        read(fd, &tmp, sizeof(tmp));
        int32_t magic = be32toh(tmp);

        read(fd, &tmp, sizeof(tmp));
        int32_t count = be32toh(tmp);

        read(fd, &tmp, sizeof(tmp));
        int32_t rows = be32toh(tmp);

        read(fd, &tmp, sizeof(tmp));
        int32_t cols = be32toh(tmp);

        printf("magic: %x, count: %u, rows: %u, cols: %u\n", magic, count, rows, cols);


        for(int i = 0; i < count; ++i) {
            for(int j = 0; j < rows*cols; ++j) {
                uint8_t c;
                read(fd, &c, sizeof(c));
                //printf("c: %d \n", (int)c);
                this->data.push_back((float)c / 255.f);
            }
        }

        close(fd);

        this->magic = magic;
        this->count = count;
        this->rows = rows;
        this->cols = cols;
    }

    int32_t magic;
    int32_t count;
    int32_t rows;
    int32_t cols;
    std::vector<float> data;
};




///// utils
std::vector<int8_t> loadLabel(std::string file) {
    auto fd = open(file.c_str(), O_RDONLY);
    assert(fd > 0);

    int32_t tmp = 0;
    read(fd, &tmp, sizeof(tmp));
    int32_t magic = be32toh(tmp);

    read(fd, &tmp, sizeof(tmp));
    int32_t count = be32toh(tmp);

    std::vector<int8_t> labels;
    for(int i=0; i<count; ++i) {
        int8_t label;
        read(fd, &label, sizeof(label));
        labels.emplace_back(label);
    }
    printf("magic: %x, count: %d\n", magic, count);
    close(fd);
    return labels;
}






#endif //NUMBERCLASSIFIER_IMAGEDATA_H
