//
// Created by 171153 on 2024/8/6.
//

#include "../include/TemporalActionDetection.h"
#include "../include/utils.h"
#include <numeric>
#include <algorithm>
#include <memory>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <tbb/parallel_for.h>


TAD::TAD(const string& modelpath,const int _origin_h, const int _origin_w, const float nms_thresh_, const float conf_thresh_, float action_thresh, float keypoint_thresh)
{
    // creat handle
    BMNNHandlePtr handle = make_shared<BMNNHandle>(0);

    mBMContext = make_shared<BMNNContext>(handle, modelpath.c_str());


    // 1. get network
    mBMNetwork = mBMContext->network(0);

    auto m_input_tensor = mBMNetwork->inputTensor(0);
    auto m_output_tensor = mBMNetwork->outputTensor(0);
    bmrt_tensor(&bmInputTensor,
                mBMContext->bmrt(),
                m_input_tensor->get_dtype(),
                *m_input_tensor->get_shape());
    m_input_tensor->set_device_mem(&bmInputTensor.device_mem);

    this->inpBatchSize = m_input_tensor->get_shape()->dims[0];
    this->len_clip = m_input_tensor->get_shape()->dims[2];
    this->inpHeight = m_input_tensor->get_shape()->dims[3];
    this->inpWidth = m_input_tensor->get_shape()->dims[4];
    this->confThresh = conf_thresh_;
    this->nmsThresh = nms_thresh_;
    this->originHeight = _origin_h;
    this->originWidth = _origin_w;
    this->boxSize = m_output_tensor->get_shape()->dims[1];
    this->clsSize = m_output_tensor->get_shape()->dims[2];
    actionThresh = action_thresh;
    keypointThresh = keypoint_thresh;


}

/*void TAD::preprocess(vector<Mat> video_clip)
{
    const int image_area = this->inpHeight * this->inpWidth;
    this->input_tensor.resize(1 * 3 * this->len_clip * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    const int chn_area = this->len_clip * image_area;
    for (int i = 0; i < this->len_clip; i++)
    {
        Mat resizeimg;
        resize(video_clip[i], resizeimg, cv::Size(this->inpWidth, this->inpHeight));
        resizeimg.convertTo(resizeimg, CV_32FC3);
        vector<cv::Mat> bgrChannels(3);
        split(resizeimg, bgrChannels);

        memcpy(this->input_tensor.data() + i * image_area, (float *)bgrChannels[0].data, single_chn_size);
        memcpy(this->input_tensor.data() + chn_area + i * image_area, (float *)bgrChannels[1].data, single_chn_size);
        memcpy(this->input_tensor.data() + 2 * chn_area + i * image_area, (float *)bgrChannels[2].data, single_chn_size);
    }
}*/

Mat TAD::pre_process_mat(Mat& input)
{
    auto resizeimg = letterbox(input, this->inpHeight, this->inpWidth);
    resizeimg.convertTo(resizeimg, CV_32FC3);
    return resizeimg;
}

std::vector<float> TAD::pre_process(std::vector<Mat>& mats)
{
    vector<float> input_tensor;
    const int image_area = this->inpHeight * this->inpWidth;
    input_tensor.resize(batchSize * this->len_clip * 3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    const int chn_area = this->len_clip * image_area;
    for(int i = 0; i < len_clip; i ++){
        vector<cv::Mat> bgrChannels(3);
        split(mats[i], bgrChannels);
        memcpy(input_tensor.data() + 0 * chn_area + i * image_area, (float *)bgrChannels[0].data, single_chn_size);
        memcpy(input_tensor.data() + 1 * chn_area + i * image_area, (float *)bgrChannels[1].data, single_chn_size);
        memcpy(input_tensor.data() + 2 * chn_area + i * image_area, (float *)bgrChannels[2].data, single_chn_size);
    }
    return input_tensor;

}

CommonResultPose TAD::detect(CommonResultPose& input)
{
    bm_memcpy_s2d(mBMContext->handle(), bmInputTensor.device_mem, input.float_vector.data());
    mBMNetwork->forward();
    const auto outputTensor = mBMNetwork->outputTensor(0);
    const auto pred = outputTensor->get_cpu_data();
    int size = this->batchSize * outputTensor->get_shape()->dims[1] * outputTensor->get_shape()->dims[2];
    vector<float> output(size);
    memcpy(output.data(), pred, size * sizeof(float));
    input.float_vector = output;
    return input;

}

CommonResultPose TAD::post_process(CommonResultPose& input)
{
    vector<ObjectPose> _object_poses;
    for(int i = 0; i < boxSize; i++)
    {
        if (input.float_vector[i * clsSize + 4] < confThresh) continue;
        ObjectPose op;
        op.rect = cv::Rect(input.float_vector[i * clsSize + 0], input.float_vector[i * clsSize + 1],
            input.float_vector[i * clsSize + 2], input.float_vector[i * clsSize + 3]);
        op.label = 0;
        for(int j = 5; j < clsSize + 5;j ++)
        {
            op.action_prob.emplace_back(input.float_vector[i * clsSize + j]);
        }
        _object_poses.emplace_back(op);
    }
    auto keep_indx = multiclass_nms_class_agnostic(_object_poses, nmsThresh);
    vector<ObjectPose> object_poses;
    for(const int& i : keep_indx)
    {
        object_poses.emplace_back(_object_poses[i]);
    }

    input.object_poses = object_poses;
    return input;
}


CommonResultPose TAD::vis(CommonResultPose& input)
{
    int baseLine = 0;
    int fontFace = cv::FONT_HERSHEY_SIMPLEX;
    double fontScale = 0.8;
    cv::Scalar color(255, 255, 255);
    int thickness = 2;


    const std::vector<std::vector<unsigned int>> KPS_COLORS =
    { {0,   255, 0},
      {0,   255, 0},
      {0,   255, 0},
      {0,   255, 0},
      {0,   255, 0},
      {255, 128, 0},
      {255, 128, 0},
      {255, 128, 0},
      {255, 128, 0},
      {255, 128, 0},
      {255, 128, 0},
      {51,  153, 255},
      {51,  153, 255},
      {51,  153, 255},
      {51,  153, 255},
      {51,  153, 255},
      {51,  153, 255} };

    const std::vector<std::vector<unsigned int>> SKELETON = { {16, 14},
                                                              {14, 12},
                                                              {17, 15},
                                                              {15, 13},
                                                              {12, 13},
                                                              {6,  12},
                                                              {7,  13},
                                                              {6,  7},
                                                              {6,  8},
                                                              {7,  9},
                                                              {8,  10},
                                                              {9,  11},
                                                              {2,  3},
                                                              {1,  2},
                                                              {1,  3},
                                                              {2,  4},
                                                              {3,  5},
                                                              {4,  6},
                                                              {5,  7} };

    const std::vector<std::vector<unsigned int>> LIMB_COLORS = { {51,  153, 255},
                                                                 {51,  153, 255},
                                                                 {51,  153, 255},
                                                                 {51,  153, 255},
                                                                 {255, 51,  255},
                                                                 {255, 51,  255},
                                                                 {255, 51,  255},
                                                                 {255, 128, 0},
                                                                 {255, 128, 0},
                                                                 {255, 128, 0},
                                                                 {255, 128, 0},
                                                                 {255, 128, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0} };

    const std::vector<int> keep_index = {7, 10, 11, 53, 66, 67, 68, 78, 77, 14};


    cv::Mat res = input.origin_mat.clone();

    for (auto& obj : input.track_vector)
    {
        cv::rectangle(res, cv::Rect(static_cast<int>(obj->getRect().x()), static_cast<int>(obj->getRect().y()),
            static_cast<int>(obj->getRect().width()), static_cast<int>(obj->getRect().height())), { 0, 0, 255 }, 2);
        string text = format("%s %.1f%% \n", "person", obj->getScore() * 100);
        // if (obj.label != 0) {
        if (false) {
            text += "infos: \n";
            std::vector<int> right_eyes = { 0, 2, 4, 6, 8, 10};
            std::vector<int> left_eyes = { 1, 3, 5, 7, 9, 11};
            std::vector<int> mouth = { 12, 14, 15, 13, 17, 16};

            float EAR = (euclidean_distance(obj->kps[right_eyes[1] * 3 + 0],
                obj->kps[right_eyes[5] * 3 + 0], obj->kps[right_eyes[1] * 3 + 1],
                obj->kps[right_eyes[5] * 3 + 1]) + euclidean_distance(obj->kps[right_eyes[2] * 3 + 0],
                    obj->kps[right_eyes[4] * 3 + 0], obj->kps[right_eyes[2] * 3 + 1],
                    obj->kps[right_eyes[4] * 3 + 1])) / (2 * euclidean_distance(obj->kps[right_eyes[0] * 3 + 0],
                        obj->kps[right_eyes[3] * 3 + 0], obj->kps[right_eyes[0] * 3 + 1],
                        obj->kps[right_eyes[3] * 3 + 1]));
            text += format("RIGHT: %.2f\n", EAR);


            EAR = (euclidean_distance(obj->kps[left_eyes[1] * 3 + 0],
                obj->kps[left_eyes[5] * 3 + 0], obj->kps[left_eyes[1] * 3 + 1],
                obj->kps[left_eyes[5] * 3 + 1]) + euclidean_distance(obj->kps[left_eyes[2] * 3 + 0],
                    obj->kps[left_eyes[4] * 3 + 0], obj->kps[left_eyes[2] * 3 + 1],
                    obj->kps[left_eyes[4] * 3 + 1])) / (2 * euclidean_distance(obj->kps[left_eyes[0] * 3 + 0],
                        obj->kps[left_eyes[3] * 3 + 0], obj->kps[left_eyes[0] * 3 + 1],
                        obj->kps[left_eyes[3] * 3 + 1]));
            text += format("LEFT: %.2f\n", EAR);


            EAR = (euclidean_distance(obj->kps[mouth[1] * 3 + 0],
                obj->kps[mouth[5] * 3 + 0], obj->kps[mouth[1] * 3 + 1],
                obj->kps[mouth[5] * 3 + 1]) + euclidean_distance(obj->kps[mouth[2] * 3 + 0],
                    obj->kps[mouth[4] * 3 + 0], obj->kps[mouth[2] * 3 + 1],
                    obj->kps[mouth[4] * 3 + 1])) / (2 * euclidean_distance(obj->kps[mouth[0] * 3 + 0],
                        obj->kps[mouth[3] * 3 + 0], obj->kps[mouth[0] * 3 + 1],
                        obj->kps[mouth[3] * 3 + 1]));
            text += format("MOUTH: %.2f", EAR);

            // cv::Size label_size = cv::getTextSize(face_text, cv::FONT_HERSHEY_SIMPLEX,
            //     0.4, 1, &baseLine);
            //
            // cv::putText(res, face_text, cv::Point(x, y - label_size.height),
            //     cv::FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 }, 1);

        }else
        {
            text += format("track id: %u \n", obj->getTrackId());

            for (int i = 0; i < obj->getActions().size(); i ++)
            {
                if(obj->getActions()[i] > actionThresh && count(keep_index.begin(), keep_index.end(), i))
                {
                    text += format("%s:%.1f%% \n",labels[i], obj->getActions()[i] * 100);
                }
            }

        }

        int x = (int)obj->getRect().x();
        int y = (int)obj->getRect().y() + 1;

        if (y > res.rows)
            y = res.rows;

        std::istringstream iss(text);
        std::string line;
        while (std::getline(iss, line)) {
            cv::putText(res, line, cv::Point(x, y), fontFace, fontScale, color, thickness);
            // 计算下一行的 y 坐标，假设行高等于文本的高度加上 baseline
            if (y + (cv::getTextSize(line, fontFace, fontScale, thickness, &baseLine).height + baseLine) <= res.rows)
            {
                y += cv::getTextSize(line, fontFace, fontScale, thickness, &baseLine).height + baseLine;
            }
        }


        std::vector<float> kps = obj->kps;
        if (!kps.empty())
        {
            for (int k = 0; k < 133; k++)
            {
                int kps_x= std::round(kps[k * 3]);
                int kps_y=std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > keypointThresh)
                {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[0][0], KPS_COLORS[0][1], KPS_COLORS[0][2]);
                    cv::circle(res, { kps_x, kps_y }, 0, kps_color, 5);
                }

            }
        }

    }
    input.processed_mat = res;

    return input;
}