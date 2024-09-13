#include "../include/featuretensor.h"
#include <fstream>

//using namespace nvinfer1;

#define INPUTSTREAM_SIZE (maxBatchSize*3*imgShape.area())
#define OUTPUTSTREAM_SIZE (maxBatchSize*featureDim)
using namespace deep_sort;
FeatureTensor::FeatureTensor(const int maxBatchSize, const cv::Size imgShape, const int featureDim, int gpuID) 
        : maxBatchSize(maxBatchSize), imgShape(imgShape), featureDim(featureDim), 
        inputName("input"), outputName("output") {
    //cudaSetDevice(gpuID);
    //this->gLogger = gLogger;
    //runtime = nullptr;
    //engine = nullptr;
    //context = nullptr; 

    means[0] = 0.485, means[1] = 0.456, means[2] = 0.406;
    std[0] = 0.229, std[1] = 0.224, std[2] = 0.225;

    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0));
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));
    char* input_name = "input";
    input_names.emplace_back(input_name);
    char* output_name = "output";
    output_names.emplace_back(output_name);
    vector<int64_t> input_vector{1,3,256,128};
    vector<int64_t> output_vector{1,512};
    input_node_dims.emplace_back(input_vector);
    output_node_dims.emplace_back(output_vector);
    initFlag = false;
}

FeatureTensor::~FeatureTensor() {
    if (initFlag) {
        // cudaStreamSynchronize(cudaStream);
        //cudaStreamDestroy(cudaStream);
        //cudaFree(buffers[inputIndex]);
        //cudaFree(buffers[outputIndex]);
    }
}

bool FeatureTensor::getRectsFeature(const cv::Mat& img, DETECTIONS& det) {
    std::vector<cv::Mat> mats;
    for (auto& dbox : det) {
        cv::Rect rect = cv::Rect(int(dbox.tlwh(0)), int(dbox.tlwh(1)),
                                 int(dbox.tlwh(2)), int(dbox.tlwh(3)));
        // rect.x -= (rect.height * 0.5 - rect.width) * 0.5;
        // rect.width = rect.height * 0.5;
        // rect.x = (rect.x >= 0 ? rect.x : 0);
        // rect.y = (rect.y >= 0 ? rect.y : 0);
        // rect.width = (rect.x + rect.width <= img.cols ? rect.width : (img.cols - rect.x));
        // rect.height = (rect.y + rect.height <= img.rows ? rect.height : (img.rows - rect.y));
        cv::Mat tempMat = img(rect).clone();
        // cv::imwrite("mat.jpg",tempMat);
        cv::resize(tempMat, tempMat, imgShape);
        mats.push_back(tempMat);
    }
    const auto* out = doInference(mats);
    // decode output to det
    int i = 0;
    for (DETECTION_ROW& dbox : det) {
        for (int j = 0; j < featureDim; ++j) {
            dbox.feature[j] = out[i * featureDim + j];
        }
        i++;
    }


    return true;
}

bool FeatureTensor::getRectsFeature(DETECTIONS& det) {
    return true;
}

void FeatureTensor::loadOnnx(std::string onnxPath) {
    //dnn_engine = new cv::dnn::Net();

    //dnn_engine.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    //dnn_engine.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    //static auto _engine = cv::dnn::readNetFromONNX(onnxPath);
    ort_session = new Session(env, onnxPath.c_str(), sessionOptions); ////linux



}

//int FeatureTensor::getResult(float*& buffer) {
//    if (buffer != nullptr)
//        delete buffer;
//    int curStreamSize = curBatchSize*featureDim;
//    buffer = new float[curStreamSize];
//    for (int i = 0; i < curStreamSize; ++i) {
//        buffer[i] = outputBuffer[i];
//    }
//    return curStreamSize;
//}


cv::Mat FeatureTensor::preprocess(const cv::Mat& video_mat)
{

    cv::Mat resizeimg;
    resize(video_mat, resizeimg, cv::Size(this->input_node_dims[0][3], this->input_node_dims[0][2]));
    resizeimg.convertTo(resizeimg, CV_32FC3, 1.0 / 255);
    // cvtColor(resizeimg, resizeimg, cv::COLOR_BGR2RGB);
    // cv::imwrite("1.jpg",resizeimg);
    return resizeimg;
}


const float* FeatureTensor::doInference(vector<cv::Mat>& imgMats) {
    //cudaMemcpyAsync(buffers[inputIndex], inputBuffer, inputStreamSize * sizeof(float), cudaMemcpyHostToDevice, cudaStream);
    //Dims4 inputDims{curBatchSize, 3, imgShape.height, imgShape.width};
    //context->setBindingDimensions(0, inputDims);
    //
    //context->enqueueV2(buffers, cudaStream, nullptr);
    //cudaMemcpyAsync(outputBuffer, buffers[outputIndex], outputStreamSize * sizeof(float), cudaMemcpyDeviceToHost, cudaStream);
    // cudaStreamSynchronize(cudaStream);

    vector<cv::Mat> pre_process_mats;
    for(cv::Mat& m: imgMats)
    {
        pre_process_mats.emplace_back(preprocess(m));
    }

    const int image_area = this->input_node_dims[0][2] * this->input_node_dims[0][3];
    input_node_dims[0][0] = pre_process_mats.size();
    output_node_dims[0][0] = pre_process_mats.size();
    input_tensor.resize(pre_process_mats.size() * 3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    const int chn_area = 3 * image_area;
    for (int i = 0; i < pre_process_mats.size(); i++)
    {
        vector<cv::Mat> bgrChannels(3);
        split(pre_process_mats[i], bgrChannels);
        memcpy(this->input_tensor.data() + i * chn_area + 0 * image_area, (float *)bgrChannels[0].data, single_chn_size);
        memcpy(this->input_tensor.data() + i * chn_area + 1 * image_area, (float *)bgrChannels[1].data, single_chn_size);
        memcpy(this->input_tensor.data() + i * chn_area + 2 * image_area, (float *)bgrChannels[2].data, single_chn_size);

    }
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_tensor.data(), this->input_tensor.size(), input_node_dims[0].data(), input_node_dims[0].size());
    std::chrono::system_clock::time_point start_time(std::chrono::system_clock::now());

    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());
    std::chrono::system_clock::time_point end_time(std::chrono::system_clock::now());
    std::chrono::duration<float> diff = end_time - start_time;
    std::cout << "Tracker 向前推理时间：" << diff.count() << std::endl;
    const auto *conf_preds = ort_outputs[0].GetTensorMutableData<float>();

    return conf_preds;
}


