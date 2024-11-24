#ifndef FEATURETENSOR_H
#define FEATURETENSOR_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
//#include <NvInfer.h>
//#include <NvOnnxParser.h>
#include "model.hpp"
#include "bmnn_utils.h"
#include "datatype.h"
//#include "cuda_runtime_api.h"
using std::vector;
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
        vector<float> doInference(vector<cv::Mat>& imgMats);

    private:
        void stream2det(cv::Mat stream, DETECTIONS& det);
        cv::Mat preprocess(const cv::Mat& video_mat);

    private:
        //nvinfer1::IRuntime* runtime;
        //nvinfer1::ICudaEngine* engine;
        //nvinfer1::IExecutionContext* context;
        const int maxBatchSize;
        const cv::Size imgShape;
        const int featureDim;
        vector<char *> input_names;
        vector<char *> output_names;
        vector<vector<int64_t>> input_node_dims;  // >=1 outputs
        vector<vector<int64_t>> output_node_dims; // >=1 outputs
        vector<float> input_tensor;

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


        std::shared_ptr<BMNNContext> m_bmContext;
        std::shared_ptr<BMNNNetwork> m_bmNetwork;
        std::shared_ptr<BMNNTensor>  m_input_tensor;
        std::shared_ptr<BMNNTensor>  m_output_tensor;
        bm_tensor_t bm_input_tensor;
    };
}
#endif
