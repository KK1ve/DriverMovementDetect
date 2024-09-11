//
// Created by 171153 on 2024/8/12.
//
#include "../include/InstanceSegmentation.h"
#include <iostream>
#include <opencv2/imgproc.hpp>

IS::IS(const string& modelpath, const float nms_thresh_, const float conf_thresh_)
{
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0));
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));

//    std::wstring widestr = std::wstring(modelpath.begin(), modelpath.end()); ////windows
//    ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows
    ort_session = new Session(env, modelpath.c_str(), sessionOptions); ////linux

    size_t numInputNodes = ort_session->GetInputCount();
    size_t numOutputNodes = ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    for (int i = 0; i < numInputNodes; i++)
    {
        unique_ptr<char, Ort::detail::AllocatedFree> name = ort_session->GetInputNameAllocated(i, allocator);
        char* _name = name.get();
        size_t len = strlen(_name) + 1;
        char* copy_name = new char[len];
        memcpy(copy_name, _name, len);
        input_names.push_back(copy_name);
        Ort::TypeInfo input_type_info = ort_session->GetInputTypeInfo(i);
        auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
        auto input_dims = input_tensor_info.GetShape();
        input_node_dims.push_back(input_dims);
    }
    for (int i = 0; i < numOutputNodes; i++)
    {
        unique_ptr<char, Ort::detail::AllocatedFree> name = ort_session->GetOutputNameAllocated(i, allocator);
        char* _name = name.get();
        size_t len = strlen(_name) + 1;
        char* copy_name = new char[len];
        memcpy(copy_name, _name, len);
        output_names.push_back(copy_name);
        Ort::TypeInfo output_type_info = ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        output_node_dims.push_back(output_dims);
    }
    this->inpHeight = this->input_node_dims[0][2];
    this->inpWidth = this->input_node_dims[0][3];
    this->conf_thresh = conf_thresh_;
    this->nms_thresh = nms_thresh_;
}

Mat IS::preprocess(const Mat& video_mat)
{
    Mat resizeimg;
    resize(video_mat, resizeimg, cv::Size(this->inpWidth, this->inpHeight));
    resizeimg.convertTo(resizeimg, CV_32FC3, 1/255.0);
    return resizeimg;
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
        cv::Mat mat(inpHeight, inpWidth,  CV_8UC3, cv::Scalar(0, 0, 0)); // 创建一个 CV_8U 类型的 Mat
        object_seg.label_name = labels[i];
        object_seg.label = i;
        object_seg.region = mat;
        boxes.emplace_back(object_seg);
    }
    // cout << "inpHeight: " << inpHeight << " mat_rows: " << boxes[0].region.rows << endl;
    for (int w = 0; w < inpWidth; w++)
    {
        for(int h = 0; h < inpHeight; h++)
        {
            float _max = FLT_MIN;
            int _max_index = boxes.size() - 1;
            for (int c = 0; c < boxes.size(); c ++)
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
    input_tensor.resize(1 * 3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    vector<cv::Mat> bgrChannels(3);
    split(preprocessed_mat, bgrChannels);
    memcpy(this->input_tensor.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input_tensor.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_tensor.data() + 2 * image_area, (float *)bgrChannels[2].data, single_chn_size);

    std::vector<int64_t> input_img_shape = {1, 3, this->inpHeight, this->inpWidth};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_tensor.data(), this->input_tensor.size(), input_img_shape.data(), input_img_shape.size());

    Ort::RunOptions runOptions;
    this->start_time = std::chrono::system_clock::now();
    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());
    this->end_time = std::chrono::system_clock::now();
    diff = this->end_time - this->start_time;
    cout << "向前推理时间：" << diff.count() << endl;
    diffs.emplace_back(diff.count());
    const float *pred = ort_outputs[0].GetTensorMutableData<float>();
    generate_proposal(pred, boxes);
    for(auto& obj: boxes)
    {
        // cout << video_mat.cols << " " << video_mat.rows << endl;
        resize(obj.region,  obj.region, Size(video_mat.cols, video_mat.rows));
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
