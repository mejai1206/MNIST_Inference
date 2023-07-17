#ifndef UNTITLED2_GPU_INFERENCE_H
#define UNTITLED2_GPU_INFERENCE_H

#include <memory>
class Model;
class ImageData;

void inferenceOnGPU(std::unique_ptr<Model>& model, ImageData& img, int imgIdx, int numBt);


#endif //UNTITLED2_GPU_INFERENCE_H
