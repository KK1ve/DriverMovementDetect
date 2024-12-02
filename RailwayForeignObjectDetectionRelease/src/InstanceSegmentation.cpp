//
// Created by 171153 on 2024/8/12.
//
#include "../include/InstanceSegmentation.h"
#include <iostream>
#include <utils.h>
#include <tbb/parallel_for.h>

IS::IS(const string& modelpath, int use_int8, const float nms_thresh_, const float conf_thresh_)
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
    this->conf_thresh = conf_thresh_;
    this->nms_thresh = nms_thresh_;


}


void IS::generateProposal(const float *pred, vector<ObjectSeg> &boxes)
{
    for(int i = 0; i < numClass; i ++)
    {
        ObjectSeg object_seg;
        cv::Mat mat(inpHeight, inpWidth,  CV_8UC3, cv::Scalar(0, 0, 0));
        object_seg.label_name = labels[i];
        object_seg.label = i;
        object_seg.region = mat;
        boxes.emplace_back(object_seg);
    }
    // cout << "inpHeight: " << inpHeight << " mat_rows: " << boxes[0].region.rows << endl;
    for(int w = 0; w < inpWidth; w ++ ){
        for(int h = 0; h < inpHeight; h++)
        {
            float _max = -1000;
            int _max_index = 2;
            for (int c = 0; c < numClass; c ++)
            {
                int index = c * inpHeight * inpWidth + h * inpWidth + w;

                if (pred[index] > _max)
                {
                    _max = pred[index];
                    _max_index = c;
                }
            }
            // cout << "w: " << w << " h: " << h << endl;
            boxes[_max_index].region.at<cv::Vec3b>(h, w)[0] = matColos[_max_index][2]; // Blue
            boxes[_max_index].region.at<cv::Vec3b>(h, w)[1] = matColos[_max_index][1];   // Green
            boxes[_max_index].region.at<cv::Vec3b>(h, w)[2] = matColos[_max_index][0];   // Red
        }
    }
    /*cout << "width: " << boxes[0].region.cols << " height: " << boxes[0].region.rows << endl;
    imshow("123",boxes[0].region);
    waitKey(1);*/
    // cout << endl;




}

std::tuple<unsigned long, std::vector<float>, Mat> IS::pre_process(std::tuple<unsigned long, Mat>& mat)
{
    const int image_area = this->inpHeight * this->inpWidth;
    vector<cv::Mat> bgrChannels(3);
    Mat result_mat = letterbox(std::get<1>(mat), this->inpHeight, this->inpWidth);
    result_mat.convertTo(result_mat, CV_32FC3, 1.0 / 255);
    vector<float> input_tensor;
    input_tensor.resize(1 * 3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    split(result_mat, bgrChannels);
    memcpy(input_tensor.data(), bgrChannels[0].data, single_chn_size);
    memcpy(input_tensor.data() + image_area, bgrChannels[1].data, single_chn_size);
    memcpy(input_tensor.data() + 2 * image_area, bgrChannels[2].data, single_chn_size);
    std::tuple<unsigned long, std::vector<float>, Mat> return_result(std::get<0>(mat), input_tensor, std::get<1>(mat));
    return return_result;
}

std::tuple<unsigned long, std::shared_ptr<BMNNTensor>, Mat> IS::detect(std::tuple<unsigned long, std::vector<float>, Mat>& input_vector)
{
    bm_memcpy_s2d(mBMContext->handle(), bmInputTensor.device_mem, &std::get<1>(input_vector)[0]);
    mBMNetwork->forward();
    std::shared_ptr<BMNNTensor> outputTensor = mBMNetwork->outputTensor(0);
    std::tuple<unsigned long, std::shared_ptr<BMNNTensor>, Mat> return_result(std::get<0>(input_vector), outputTensor, std::get<2>(input_vector));
    return return_result;
}

std::tuple<unsigned long, vector<ObjectSeg>, Mat> IS::post_process(std::tuple<unsigned long, std::shared_ptr<BMNNTensor>, Mat>& pred)
{
    std::vector<ObjectSeg> object_segs;
    generateProposal(std::get<1>(pred)->get_cpu_data(), object_segs);
    for(auto& obj: object_segs)
    {
        obj.region = un_letterbox(obj.region, std::get<2>(pred).rows, std::get<2>(pred).cols);
    }
    std::tuple<unsigned long, vector<ObjectSeg>, Mat> return_result(std::get<0>(pred), object_segs, std::get<2>(pred));
    return return_result;
}
std::tuple<unsigned long, Mat, Mat> IS::vis(std::tuple<unsigned long, vector<ObjectSeg>, Mat>& boxes)
{
    Mat result_mat = std::get<2>(boxes).clone();
    for(auto& obj : std::get<1>(boxes))
    {
        addWeighted(result_mat, 1, obj.region, 1, 1, result_mat);
    }
    std::tuple<unsigned long, Mat, Mat> return_result(std::get<0>(boxes), result_mat, std::get<2>(boxes));
    return return_result;
}
