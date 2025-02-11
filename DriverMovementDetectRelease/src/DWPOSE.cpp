//
// Created by 171153 on 2024/10/17.
//

#include "../include/DWPOSE.h"

#include <iostream>
#include <utils.h>

DWPOSE::DWPOSE(const string& modelpath,int _originHeight, int _originWidth)
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
    this->inpBatchSize = m_input_tensor->get_shape()->dims[0];
    this->inpHeight = m_input_tensor->get_shape()->dims[2];
    this->inpWidth = m_input_tensor->get_shape()->dims[3];
    auto m_output_tensor = mBMNetwork->outputTensor(0);
    this->keypointsCount = m_output_tensor->get_shape()->dims[1];
    originHeight = _originHeight;
    originWidth = _originWidth;
}


CommonResultPose DWPOSE::pre_process(CommonResultPose& input)
{

    cv::Scalar means(this->means[0], this->means[1], this->means[2]);
    cv::Mat mean_mat(inpHeight, inpWidth, CV_32FC3, means);
    vector<cv::Mat> bgrChannels(3);
    vector<float> input_tensor(input.track_vector.size() * inpHeight * inpWidth * 3);

    const int image_area = this->inpHeight * this->inpWidth;

    const int batch_size = image_area * 3;
    size_t single_chn_size = image_area * sizeof(float);
    for(int i = 0; i < input.track_vector.size(); i ++)
    {
        Mat result_mat = letterbox(input.origin_mat(cv::Rect(static_cast<int>(input.track_vector[i]->getRect().x()), static_cast<int>(input.track_vector[i]->getRect().y()),
            static_cast<int>(input.track_vector[i]->getRect().width()), static_cast<int>(input.track_vector[i]->getRect().height()))), this->inpHeight, this->inpWidth);
        split(result_mat, bgrChannels);
        for (int j = 0; j < 3; j ++)
        {
            bgrChannels[j] *= this->stds[j];
        }
        memcpy(input_tensor.data() + i * batch_size, bgrChannels[0].data, single_chn_size);
        memcpy(input_tensor.data() + i * batch_size + image_area, bgrChannels[1].data, single_chn_size);
        memcpy(input_tensor.data() + i * batch_size + 2 * image_area, bgrChannels[2].data, single_chn_size);
    }

    input.float_vector = input_tensor;

    return input;


}

CommonResultPose DWPOSE::detect(CommonResultPose& input)
{

    vector<float> out_tensort(input.track_vector.size() * 3 * keypointsCount);

    const int image_area = this->inpHeight * this->inpWidth;

    const int batch_size = image_area * 3;

    int b = -1; //index
    int i = -1; //current batch index

    vector<float> input_tensor(inpBatchSize * 3 * inpHeight * inpWidth);


    for(int k = 0; k < input.track_vector.size(); k ++)
    {
        i += 1;
        b += 1;

        std::copy(input.float_vector.begin() + k * batch_size, input.float_vector.begin() + (k + 1) * batch_size, input_tensor.begin() + k * batch_size);

        if (b >= inpBatchSize - 1|| i == input.track_vector.size() - 1)
        {
            for (int j = 1; j <= inpBatchSize - 1 - b; j ++)
            {
                std::copy(input_tensor.begin(), input_tensor.begin() + batch_size, input_tensor.begin() + (inpBatchSize - j) * batch_size);
            }
            bm_memcpy_s2d(mBMContext->handle(), bmInputTensor.device_mem, &input_tensor[0]);
            mBMNetwork->forward();
            auto x_locs_ptr = mBMNetwork->outputTensor(0);
            auto y_locs_ptr = mBMNetwork->outputTensor(1);
            auto vals_ptr = mBMNetwork->outputTensor(2);

            auto x_locs = x_locs_ptr->get_cpu_data();
            auto y_locs = y_locs_ptr->get_cpu_data();
            auto vals = vals_ptr->get_cpu_data();

            for (int _i = 0; _i <= b; _i++)
            {
                std::transform(x_locs, x_locs + (_i + 1) * keypointsCount, out_tensort.begin() + (i - b + _i) * 3 * keypointsCount + 0 * keypointsCount,
                    [](float x) {
                    return x / 2;
                });
                std::transform(y_locs, y_locs + (_i + 1) * keypointsCount, out_tensort.begin() + (i - b + _i) * 3 * keypointsCount + 1 * keypointsCount,
                    [](float x) {
                    return x / 2;
                });

                std::copy(vals, vals + (_i + 1) * keypointsCount, out_tensort.begin() + (i - b + _i) * 3 * keypointsCount + 2 * keypointsCount);
            }
            b = -1;
        }
    }
    input.float_vector = out_tensort;

    return input;


}


CommonResultPose DWPOSE::post_process(CommonResultPose& input)
{

    for (int n = 0; n < input.track_vector.size(); n ++){
        const float scale = min(float(inpHeight) / float(input.track_vector[n]->getRect().height()), float(inpWidth) / float(input.track_vector[n]->getRect().width()));
        const int padd_w = round((float(inpWidth) - float(input.track_vector[n]->getRect().width()) * scale) / 2.0f);
        const int padd_h = round((float(inpHeight) - float(input.track_vector[n]->getRect().height()) * scale) / 2.0f);
        for (int i = 0 ; i < keypointsCount; i ++)
        {
            input.track_vector[n]->kps.emplace_back((input.float_vector[n * keypointsCount * 3 + 0 * keypointsCount + i] - padd_w) / scale + input.track_vector[n]->getRect().x());
            input.track_vector[n]->kps.emplace_back((input.float_vector[n * keypointsCount * 3 + 1 * keypointsCount + i] - padd_h) / scale + input.track_vector[n]->getRect().y());
            input.track_vector[n]->kps.emplace_back(input.float_vector[n * keypointsCount * 3 + 2 * keypointsCount + i]);
        }
    }

    return input;

}


