//
// Created by 171153 on 2024/10/17.
//

#include "../include/DWPOSE.h"
#include "../include/DWPOSE.h"

#include <iostream>
#include <utils.h>

DWPOSE::DWPOSE(const string& modelpath, int use_int8, const float nms_thresh_, const float conf_thresh_)
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
    this->inpHeight = m_input_tensor->get_shape()->dims[2];
    this->inpWidth = m_input_tensor->get_shape()->dims[3];
    this->conf_thresh = conf_thresh_;
    this->nms_thresh = nms_thresh_;


}

Mat DWPOSE::preprocess(const Mat& video_mat)
{
    Mat result_mat = letterbox(video_mat, this->inpHeight, this->inpWidth);
    result_mat.convertTo(result_mat, CV_32FC3);
    return result_mat;
}

void DWPOSE::generate_proposal(const vector<float>& keypoints, vector<ObjectPose> &boxes, int origin_w, int origin_h, int precessed_w, int precessed_h)
{
    float r = std::min(float(precessed_w) / origin_w, float(precessed_h) / origin_h);
    int inside_w = round(origin_w * r);
    int inside_h = round(precessed_w * r);

    int padd_w = (precessed_w - inside_w) / 2;
    int padd_h = (precessed_h - inside_h) / 2;

    for (int n = 0; n < boxes.size(); n ++){
        std::vector<float> kp;
        for (int i = 0 ; i < num_keypoint; i ++)
        {
            kp.emplace_back(((keypoints[n * num_keypoint * 3 + i * 3 + 0] / 2) - padd_h)/ r);
            kp.emplace_back(((keypoints[n * num_keypoint * 3 + i * 3 + 1] / 2) - padd_w)/ r);
            kp.emplace_back(keypoints[n * num_keypoint * 3 + i * 3 + 2]);

        }
        boxes[n].kps = kp;
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
    cout << m_input_tensor->get_shape()->dims[0] << endl;
    input_tensor.resize(m_input_tensor->get_shape()->dims[0] * 3 * image_area);

    vector<float> out_tensort;

    size_t single_chn_size = image_area * sizeof(float);
    cv::Scalar mean(this->means[0], this->means[1], this->means[2]);
    cv::Mat mean_mat(m_input_tensor->get_shape()->dims[2], m_input_tensor->get_shape()->dims[3], CV_32FC3, mean);

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

        if (b >= m_input_tensor->get_shape()->dims[0] - 1|| i == track_imgs.size() - 1)
        {
            for (int j = 1; j <= m_input_tensor->get_shape()->dims[0] - 1 - b; j ++)
            {
                std::copy(input_tensor.begin(), input_tensor.begin() + batch_size, input_tensor.begin() + (m_input_tensor->get_shape()->dims[0] - j));
            }
            bm_memcpy_s2d(m_bmContext->handle(), bm_input_tensor.device_mem, &input_tensor[0]);
            m_bmNetwork->forward();

            auto simcc_x = m_bmNetwork->outputTensor(0)->get_cpu_data();
            for(int k = 0; k < 133; k ++)
            {
                cout << simcc_x[k] << "     ";
            }
            cout << endl;
            const auto simcc_y = m_bmNetwork->outputTensor(1)->get_cpu_data();
            // std::copy(keypoints, keypoints + m_input_tensor->get_shape()->dims[0] * output_single_batch_size,
            //     out_tensort.begin() + i * output_single_batch_size);

            b = -1;
        }

    }

    generate_proposal(out_tensort, boxes, track_imgs.begin()->second.cols, track_imgs.begin()->second.rows, m_input_tensor->get_shape()->dims[3], m_input_tensor->get_shape()->dims[2]);

}


Mat DWPOSE::vis(const Mat& frame, vector<ObjectPose>& boxes)
{
    Mat result_mat = frame.clone();

    // imshow("123",result_mat);
    // waitKey(1);
    return result_mat;

}

