//
// Created by 171153 on 2024/8/12.
//
#include "../include/InstanceSegmentation.h"
#include <iostream>
#include <utils.h>
#include <tbb/parallel_for.h>
#include <tbb/task_arena.h>

IS::IS(const string& modelpath, const float nms_thresh_, const float conf_thresh_)
{
    // creat handle
    BMNNHandlePtr handle = make_shared<BMNNHandle>(0);

    mBMContext = make_shared<BMNNContext>(handle, modelpath.c_str());


    // 1. get network
    mBMNetwork = mBMContext->network(0);

    auto m_input_tensor = mBMNetwork->inputTensor(0);
    bmrt_tensor(&bmInputTensor,
                mBMContext->bmrt(),
                m_input_tensor->get_dtype(),
                *m_input_tensor->get_shape());
    m_input_tensor->set_device_mem(&bmInputTensor.device_mem);
    batchSize = m_input_tensor->get_shape()->dims[0];
    this->inpHeight = m_input_tensor->get_shape()->dims[2];
    this->inpWidth = m_input_tensor->get_shape()->dims[3];
    this->confThresh = conf_thresh_;
    this->nmsThresh = nms_thresh_;

    image_area = this->inpHeight * this->inpWidth;


}


void IS::generateProposal(const vector<float>& pred, CommonResultSeg &input)
{
    cv::Mat mat(inpHeight, inpWidth,  CV_8UC3, cv::Scalar(0, 0, 0));
    tbb::parallel_for(0, inpWidth, [&](int w)
    {
        for(int h = 0; h < inpHeight; h++)
        {
            const int index = h * inpWidth + w;
            // cout << "w: " << w << " h: " << h << endl;
            mat.at<cv::Vec3b>(h, w)[0] = matColos[static_cast<int>(pred[index])][2]; // Blue
            mat.at<cv::Vec3b>(h, w)[1] = matColos[static_cast<int>(pred[index])][1];   // Green
            mat.at<cv::Vec3b>(h, w)[2] = matColos[static_cast<int>(pred[index])][0];   // Red
        }
    });
    input.processed_mat = mat;
}

CommonResultSeg IS::pre_process(CommonResultSeg& input)
{

    vector<cv::Mat> bgrChannels(3);
    Mat result_mat = letterbox(input.origin_mat, this->inpHeight, this->inpWidth);
    result_mat.convertTo(result_mat, CV_32FC3, 1.0 / 255);
    vector<float> input_tensor(batchSize * 3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    split(result_mat, bgrChannels);
    tbb::parallel_for(0, 3, [&](int i)
    {
        memcpy(input_tensor.data() + i * image_area, bgrChannels[i].data, single_chn_size);
    });
    input.float_vector = input_tensor;
    return input;
}

CommonResultSeg IS::detect(CommonResultSeg& input)
{
    bm_memcpy_s2d(mBMContext->handle(), bmInputTensor.device_mem, input.float_vector.data());
    mBMNetwork->forward();
    const auto outputTensor = mBMNetwork->outputTensor(0);
    const auto pred = outputTensor->get_cpu_data();
    int size = this->batchSize * this->inpHeight * this->inpWidth;
    vector<float> output(size);
    memcpy(output.data(), pred, size * sizeof(float));
    input.float_vector = output;
    return input;
}

CommonResultSeg IS::post_process(CommonResultSeg& input)
{
    generateProposal(input.float_vector, input);
    input.processed_mat = un_letterbox(input.processed_mat, input.origin_mat.rows, input.origin_mat.cols);
    return input;
}
CommonResultSeg IS::vis(CommonResultSeg& input)
{
    Mat result_mat = input.origin_mat.clone();
    addWeighted(result_mat, 1, input.processed_mat, 1, 1, result_mat);
    input.processed_mat = result_mat;
    return input;
}
