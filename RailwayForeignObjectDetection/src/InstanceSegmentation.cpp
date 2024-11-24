//
// Created by 171153 on 2024/8/12.
//
#include "../include/InstanceSegmentation.h"
#include <iostream>

IS::IS(const string& modelpath, int use_int8, const float nms_thresh_, const float conf_thresh_)
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
    this->use_int8 = use_int8;


}

Mat IS::preprocess(const Mat& video_mat)
{
    Mat result_mat = letterbox(video_mat, this->inpHeight, this->inpWidth);
    if (this->use_int8)
    {
        result_mat.convertTo(result_mat, CV_32FC3);
    }else
    {
        result_mat.convertTo(result_mat, CV_32FC3, 1.0 / 255);

    }
    return result_mat;
}
void IS::generate_proposal(const float *pred, vector<ObjectSeg> &boxes)
{
    /*vector<vector<vector<float>>> temp;
    for (int c = 0; c < boxes.size(); c ++)
    {
        for (int h = 0; h < inpHeight; h ++)
        {
            for (int w = 0; w < inpWidth; w ++)
            {
                temp[c][w][h] = pred[c * inpHeight * inpWidth + h * inpWidth + w];
            }
        }
    }*/
    for(int i = 0; i < num_class; i ++)
    {
        ObjectSeg object_seg;
        cv::Mat mat(inpHeight, inpWidth,  CV_8UC3, cv::Scalar(0, 0, 0));
        object_seg.label_name = labels[i];
        object_seg.label = i;
        object_seg.region = mat;
        boxes.emplace_back(object_seg);
    }
    // cout << "inpHeight: " << inpHeight << " mat_rows: " << boxes[0].region.rows << endl;
#pragma omp parallel for
    for (int w = 0; w < inpWidth; w++)
    {
        for(int h = 0; h < inpHeight; h++)
        {
            float _max = -1000;
            int _max_index = 2;
            for (int c = 0; c < num_class; c ++)
            {
                int index = c * inpHeight * inpWidth + h * inpWidth + w;

                if (pred[index] > _max)
                {
                    _max = pred[index];
                    _max_index = c;
                }
            }
            // cout << "w: " << w << " h: " << h << endl;
            boxes[_max_index].region.at<cv::Vec3b>(h, w)[0] = mat_colos[_max_index][2]; // Blue
            boxes[_max_index].region.at<cv::Vec3b>(h, w)[1] = mat_colos[_max_index][1];   // Green
            boxes[_max_index].region.at<cv::Vec3b>(h, w)[2] = mat_colos[_max_index][0];   // Red
        }
    }
    /*cout << "width: " << boxes[0].region.cols << " height: " << boxes[0].region.rows << endl;
    imshow("123",boxes[0].region);
    waitKey(1);*/
    // cout << endl;




}
void IS::detect(const Mat& video_mat, vector<ObjectSeg> &boxes)
{
    Mat preprocessed_mat = preprocess(video_mat);
    const int image_area = this->inpHeight * this->inpWidth;
    vector<cv::Mat> bgrChannels(3);
    if(this->use_int8)
    {
        input_tensor_int8.resize(1 * 3 * image_area);
        size_t single_chn_size = image_area * sizeof(signed char);
        cv::Scalar mean(103.94, 116.78, 123.68);
        cv::Mat mean_mat(preprocessed_mat.rows, preprocessed_mat.cols, CV_32FC3, mean);
        cv::subtract(preprocessed_mat, mean_mat, preprocessed_mat);
        preprocessed_mat.convertTo(preprocessed_mat, CV_8SC3);
        split(preprocessed_mat, bgrChannels);
        memcpy(this->input_tensor_int8.data(), bgrChannels[0].data, single_chn_size);
        memcpy(this->input_tensor_int8.data() + image_area, bgrChannels[1].data, single_chn_size);
        memcpy(this->input_tensor_int8.data() + 2 * image_area, bgrChannels[2].data, single_chn_size);
        bm_memcpy_s2d(m_bmContext->handle(), bm_input_tensor.device_mem, &input_tensor_int8[0]);
    }else
    {
        input_tensor.resize(1 * 3 * image_area);
        size_t single_chn_size = image_area * sizeof(float);
        split(preprocessed_mat, bgrChannels);
        memcpy(this->input_tensor.data(), bgrChannels[0].data, single_chn_size);
        memcpy(this->input_tensor.data() + image_area, bgrChannels[1].data, single_chn_size);
        memcpy(this->input_tensor.data() + 2 * image_area, bgrChannels[2].data, single_chn_size);
        bm_memcpy_s2d(m_bmContext->handle(), bm_input_tensor.device_mem, &input_tensor[0]);
    }


    this->start_time = std::chrono::system_clock::now();
    m_bmNetwork->forward();
    this->end_time = std::chrono::system_clock::now();
    diff = this->end_time - this->start_time;
    cout << "向前推理时间：" << diff.count() << endl;
    diffs.emplace_back(diff.count());
    std::shared_ptr<BMNNTensor> outputTensor = m_bmNetwork->outputTensor(0);
    auto output_data = outputTensor->get_cpu_data();


    generate_proposal(output_data, boxes);
    for(auto& obj: boxes)
    {
        obj.region = un_letterbox(obj.region, video_mat.rows, video_mat.cols);
    }



}


Mat IS::vis(const Mat& frame, vector<ObjectSeg>& boxes)
{
    Mat result_mat = frame.clone();
    for(auto& obj : boxes)
    {
        addWeighted(result_mat, 1, obj.region, 1, 1, result_mat);
    }
    // imshow("123",result_mat);
    // waitKey(1);
    return result_mat;

}
