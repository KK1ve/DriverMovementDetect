//
// Created by 171153 on 2024/10/17.
//

#include "../include/DWPOSE.h"

#include <iostream>
#include <utils.h>

DWPOSE::DWPOSE(const string& modelpath,int _originHeight, int _originWidth, int use_int8, const float nms_thresh_, const float conf_thresh_)
{
    // creat handle
    BMNNHandlePtr handle = make_shared<BMNNHandle>(0);

    m_bmContext = make_shared<BMNNContext>(handle, modelpath.c_str());


    // 1. get network
    m_bmNetwork = m_bmContext->network(0);

    m_input_tensor = m_bmNetwork->inputTensor(0);
    bmrt_tensor(&bm_input_tensor,
                m_bmContext->bmrt(),
                m_input_tensor->get_dtype(),
                *m_input_tensor->get_shape());
    m_input_tensor->set_device_mem(&bm_input_tensor.device_mem);
    this->inpBatchSize = m_input_tensor->get_shape()->dims[0];
    this->inpHeight = m_input_tensor->get_shape()->dims[2];
    this->inpWidth = m_input_tensor->get_shape()->dims[3];
    this->conf_thresh = conf_thresh_;
    this->nms_thresh = nms_thresh_;
    auto m_output_tensor = m_bmNetwork->outputTensor(0);
    this->num_keypoint = m_output_tensor->get_shape()->dims[1];
    originHeight = _originHeight;
    originWidth = _originWidth;


}

Mat DWPOSE::preprocess(const Mat& video_mat)
{
    Mat result_mat = letterbox(video_mat, this->inpHeight, this->inpWidth);
    result_mat.convertTo(result_mat, CV_32FC3);
    return result_mat;
}

void DWPOSE::generate_proposal(const vector<float>& keypoints, vector<ObjectPose> &boxes)
{
    for (int n = 0; n < boxes.size(); n ++){
        const float scale = min(float(inpHeight) / float(boxes[n].rect.height), float(inpWidth) / float(boxes[n].rect.width));
        int padd_w = round((float(inpWidth) - float(boxes[n].rect.width) * scale) / 2.0f);
        int padd_h = round((float(inpHeight) - float(boxes[n].rect.height) * scale) / 2.0f);
        for (int i = 0 ; i < num_keypoint; i ++)
        {
            boxes[n].kps.emplace_back((keypoints[n * num_keypoint * 3 + 0 * num_keypoint + i] - padd_w) / scale + boxes[n].rect.x);
            boxes[n].kps.emplace_back((keypoints[n * num_keypoint * 3 + 1 * num_keypoint + i] - padd_h) / scale + boxes[n].rect.y);
            boxes[n].kps.emplace_back(keypoints[n * num_keypoint * 3 + 2 * num_keypoint + i]);
        }
    }

}


void DWPOSE::detect(const std::map<unsigned long, Mat>& track_imgs, vector<ObjectPose> &boxes)
{
    map<unsigned long, Mat> preprocessed_track_imgs;
    for(auto& item: track_imgs)
    {
        preprocessed_track_imgs[item.first] = preprocess(item.second);
    }
    const int image_area = this->inpHeight * this->inpWidth;

    const int batch_size = image_area * 3;
    input_tensor.resize(1 * 3 * image_area);

    vector<float> out_tensort;
    out_tensort.resize(track_imgs.size() * 3 * num_keypoint);

    size_t single_chn_size = image_area * sizeof(float);
    cv::Scalar mean(this->means[0], this->means[1], this->means[2]);
    cv::Mat mean_mat(inpHeight, inpWidth, CV_32FC3, mean);

    int b = -1; //index
    int i = -1; //current batch index
    for(auto& img : preprocessed_track_imgs)
    {
        i += 1;
        b += 1;
        cv::subtract(img.second, mean_mat, img.second);
        vector<cv::Mat> bgrChannels(3);
        split(img.second, bgrChannels);
        for (int j = 0; j < 3; j ++)
        {
            bgrChannels[j] *= this->stds[j];
        }
        memcpy(this->input_tensor.data() + b * batch_size, bgrChannels[0].data, single_chn_size);
        memcpy(this->input_tensor.data() + b * batch_size + image_area, bgrChannels[1].data, single_chn_size);
        memcpy(this->input_tensor.data() + b * batch_size + 2 * image_area, bgrChannels[2].data, single_chn_size);

        if (b >= inpBatchSize - 1|| i == track_imgs.size() - 1)
        {
            for (int j = 1; j <= inpBatchSize - 1 - b; j ++)
            {
                std::copy(input_tensor.begin(), input_tensor.begin() + batch_size, input_tensor.begin() + (inpBatchSize - j));
            }
            bm_memcpy_s2d(m_bmContext->handle(), bm_input_tensor.device_mem, &input_tensor[0]);
            m_bmNetwork->forward();

            auto x_locs_ptr = m_bmNetwork->outputTensor(0);
            auto y_locs_ptr = m_bmNetwork->outputTensor(1);
            auto vals_ptr = m_bmNetwork->outputTensor(2);

            auto x_locs = x_locs_ptr->get_cpu_data();
            auto y_locs = y_locs_ptr->get_cpu_data();
            auto vals = vals_ptr->get_cpu_data();

            for (int _i = 0; _i <= b; _i++)
            {
                std::transform(x_locs, x_locs + (_i + 1) * num_keypoint, out_tensort.begin() + (i - b + _i) * 3 * num_keypoint + 0 * num_keypoint,
                    [](float x) {
                    return x / 2;
                });
                std::transform(y_locs, y_locs + (_i + 1) * num_keypoint, out_tensort.begin() + (i - b + _i) * 3 * num_keypoint + 1 * num_keypoint,
                    [](float x) {
                    return x / 2;
                });

                std::copy(vals, vals + (_i + 1) * num_keypoint, out_tensort.begin() + (i - b + _i) * 3 * num_keypoint + 2 * num_keypoint);
            }

            b = -1;
        }

    }

    generate_proposal(out_tensort, boxes);

}


Mat DWPOSE::vis(const Mat& frame, vector<ObjectPose>& boxes)
{
    Mat result_mat = frame.clone();

    // imshow("123",result_mat);
    // waitKey(1);
    return result_mat;

}

