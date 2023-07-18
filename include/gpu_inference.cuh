#ifndef UNTITLED2_GPU_INFERENCE_CUH
#define UNTITLED2_GPU_INFERENCE_CUH

#include <memory>
#include <unordered_map>
#include <list>
#include <vector>

#define ASSERT_CUDA assert(cudaGetLastError() == cudaSuccess)

class Model;
class ImageData;

struct MKN {int m; int k; int n;};

class InferenceManager {
public:
    InferenceManager(Model* model, int inpSz, int numBt, const std::vector<int8_t>& labels);
    void inferenceOnGPU(ImageData& img, int imgIdx, std::vector<int8_t>& labels);
    bool checkFinish();
    int matchCount() const { return m_matchCount; }
    ~InferenceManager();

private:
    int m_inpSz = 0;
    int m_numBt = 1;
    Model* m_model = nullptr;

    std::vector<MKN> m_mkn;

    std::vector<float*> m_wBuffers;
    std::vector<float*> m_bBuffers;
    std::vector<float*> m_outBuffers;
    float* m_inpBuffer = nullptr;
    int8_t* m_labelBuffer = nullptr;
    int* m_pCnt;

    int m_matchCount = 0;
};



#endif //UNTITLED2_GPU_INFERENCE_CUH
