#ifndef FEATURETENSOR_H
#define FEATURETENSOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
//#include <NvInfer.h>
//#include <NvOnnxParser.h>
#include "model.hpp"
#include "datatype.h"
//#include "cuda_runtime_api.h"
#include "onnxruntime_cxx_api.h"
using std::vector;
using namespace Ort;
namespace deep_sort
{
    //using nvinfer1::ILogger;

    class FeatureTensor {
    public:
        //FeatureTensor(const int maxBatchSize, const cv::Size imgShape, const int featureDim, int gpuID, ILogger* gLogger);
        FeatureTensor(const int maxBatchSize, const cv::Size imgShape, const int featureDim, int gpuID);

        ~FeatureTensor();

    public:
        bool getRectsFeature(const cv::Mat& img, DETECTIONS& det);
        bool getRectsFeature(DETECTIONS& det);
        void loadOnnx(std::string onnxPath);
        //int getResult(float*& buffer);
        const float* doInference(vector<cv::Mat>& imgMats);

    private:
        void stream2det(cv::Mat stream, DETECTIONS& det);
        cv::Mat preprocess(const cv::Mat& video_mat);

    private:
        //nvinfer1::IRuntime* runtime;
        //nvinfer1::ICudaEngine* engine;
        //nvinfer1::IExecutionContext* context;
        Env env = Env(ORT_LOGGING_LEVEL_VERBOSE, "DeepSort Track");
        Session* ort_session = nullptr;
        SessionOptions sessionOptions = SessionOptions();
        const OrtApi& api = Ort::GetApi();
        const int maxBatchSize;
        const cv::Size imgShape;
        const int featureDim;
        RunOptions runOptions;
        vector<char *> input_names;
        vector<char *> output_names;
        vector<vector<int64_t>> input_node_dims;  // >=1 outputs
        vector<vector<int64_t>> output_node_dims; // >=1 outputs
        vector<float> input_tensor;
        Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    private:
        int curBatchSize;
        //const int inputStreamSize, outputStreamSize;
        bool initFlag;
        cv::Mat outputBuffer;
        //float* const inputBuffer;
        //float* const outputBuffer;
        int inputIndex, outputIndex;
        void* buffers[2];
        //cudaStream_t cudaStream;
        // BGR format
        float means[3], std[3];
        const std::string inputName, outputName;
        //ILogger* gLogger;
    };
}
#endif
