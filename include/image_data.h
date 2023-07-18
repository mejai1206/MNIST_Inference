#ifndef NUMBERCLASSIFIER_IMAGEDATA_H
#define NUMBERCLASSIFIER_IMAGEDATA_H

#include <fstream>
#include <sstream>
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

        //with transform
        for(int i = 0; i < count; ++i) {
            for(int j = 0; j < rows*cols; ++j) {
                uint8_t c;
                read(fd, &c, sizeof(c));
                this->data.push_back((((float)c/255.f) - 0.1307f) / 0.3081f);
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


#endif //NUMBERCLASSIFIER_IMAGEDATA_H
