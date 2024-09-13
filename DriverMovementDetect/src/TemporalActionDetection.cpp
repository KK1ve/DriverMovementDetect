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
static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

std::vector<int> TopKIndex(const std::vector<float> &vec, int topk)
{
    std::vector<int> topKIndex;
    topKIndex.clear();

    std::vector<size_t> vec_index(vec.size());
    std::iota(vec_index.begin(), vec_index.end(), 0);

    std::sort(vec_index.begin(), vec_index.end(), [&vec](size_t index_1, size_t index_2)
    { return vec[index_1] > vec[index_2]; });

    int k_num = std::min<int>(vec.size(), topk);

    for (int i = 0; i < k_num; ++i)
    {
        topKIndex.emplace_back(vec_index[i]);
    }

    return topKIndex;
}

int sub2ind(const int row, const int col, const int cols, const int rows)
{
    return row * cols + col;
}

void ind2sub(const int sub, const int cols, const int rows, int &row, int &col)
{
    row = sub / cols;
    col = sub % cols;
}

float GetIoU(const Bbox box1, const Bbox box2)
{
    int x1 = std::max(box1.xmin, box2.xmin);
    int y1 = std::max(box1.ymin, box2.ymin);
    int x2 = std::min(box1.xmax, box2.xmax);
    int y2 = std::min(box1.ymax, box2.ymax);
    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    float over_area = w * h;
    if (over_area == 0)
        return 0.0;
    float union_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin) + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin) - over_area;
    return over_area / union_area;
}

std::vector<int> multiclass_nms_class_agnostic(std::vector<Bbox> boxes, std::vector<float> confidences, const float nms_thresh)
{
    std::sort(confidences.begin(), confidences.end(), [&confidences](size_t index_1, size_t index_2)
    { return confidences[index_1] > confidences[index_2]; });
    const int num_box = confidences.size();
    std::vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < num_box; ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            float ovr = GetIoU(boxes[i], boxes[j]);
            if (ovr > nms_thresh)
            {
                isSuppressed[j] = true;
            }
        }
    }

    std::vector<int> keep_inds;
    for (int i = 0; i < isSuppressed.size(); i++)
    {
        if (!isSuppressed[i])
        {
            keep_inds.emplace_back(i);
        }
    }
    return keep_inds;
}

bool isZero(int num) { return num == 0; }

TAD::TAD(const string& modelpath,const int _origin_h, const int _origin_w, const float nms_thresh_, const float conf_thresh_, const int _rate)
{
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
    // ThrowOnError(OrtSessionOptionsAppendExecutionProvider_Tensorrt(sessionOptions, 0));
    ThrowOnError(OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0));

//    std::wstring widestr = std::wstring(modelpath.begin(), modelpath.end()); ////windows
//    ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows
    ort_session = new Session(env, modelpath.c_str(), sessionOptions); ////linux
    rate = _rate;
    current_rate = 0;
    for (int i = 0; i < rate ; i ++)
    {
        vector<Mat> z;
        video_clips.emplace_back(z);
    }
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
    this->len_clip = this->input_node_dims[0][2];
    this->inpHeight = this->input_node_dims[0][3];
    this->inpWidth = this->input_node_dims[0][4];
    this->conf_thresh = conf_thresh_;
    this->nms_thresh = nms_thresh_;
    this->origin_h = _origin_h;
    this->origin_w = _origin_w;

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

Mat TAD::preprocess(const Mat& video_mat)
{
    Mat resizeimg;
    resize(video_mat, resizeimg, cv::Size(this->inpHeight, this->inpWidth));
    resizeimg.convertTo(resizeimg, CV_32FC3, 1.0 / 255);
    return resizeimg;

}

void TAD::generate_proposal_one_hot(const int stride, const float *conf_pred, const float *cls_pred, const float *reg_pred, vector<Bbox> &boxes, vector<float> &det_conf, vector<vector<float>> &cls_conf)
{
    const int feat_h = (int)ceil((float)this->inpHeight / stride);
    const int feat_w = (int)ceil((float)this->inpWidth / stride);
    const int area = feat_h * feat_w;
    vector<float> conf_pred_i(area);
    for (int i = 0; i < area; i++)
    {
        conf_pred_i[i] = sigmoid(conf_pred[i]);
    }
    vector<int> topk_inds = TopKIndex(conf_pred_i, this->topk);
    int length = this->num_class;

    for (int i = 0; i < topk_inds.size(); i++)
    {
        const int ind = topk_inds[i];
        if (conf_pred_i[ind] > this->conf_thresh)
        {
            int row = 0, col = 0;
            ind2sub(ind, feat_w, feat_h, row, col);

            float cx = (col + 0.5f + reg_pred[ind * 4]) * stride;
            float cy = (row + 0.5f + reg_pred[ind * 4 + 1]) * stride;
            float w = exp(reg_pred[ind * 4 + 2]) * stride;
            float h = exp(reg_pred[ind * 4 + 3]) * stride;
            boxes.emplace_back(Bbox{int(cx - 0.5 * w), int(cy - 0.5 * h), int(cx + 0.5 * w), int(cy + 0.5 * h)});
            det_conf.emplace_back(conf_pred_i[ind]);

            vector<float> cls_conf_i(length);
            for (int j = 0; j < length; j++)
            {
                cls_conf_i[j] = sigmoid(cls_pred[ind * this->num_class + j]);
            }
            cls_conf.emplace_back(cls_conf_i);
        }
    }
}

void TAD::clear_clips_cache()
{
    multi_video_clips.clear();
}

vector<int> TAD::detect_one_hot(const Mat& input_mat, vector<Bbox> &boxes, vector<float> &det_conf, vector<vector<float>> &cls_conf)
{
    Mat preprocessed_mat = preprocess(input_mat);
    if(video_clips[current_rate].empty())
    {
        for (int i = 0; i < len_clip; i ++)
        {
            video_clips[current_rate].emplace_back(preprocessed_mat);

        }
    }else
    {
        video_clips[current_rate].erase(video_clips[current_rate].begin());
        video_clips[current_rate].emplace_back(preprocessed_mat);

    }
    const int image_area = this->inpHeight * this->inpWidth;
    input_tensor.resize(1 * 3 * this->len_clip * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    const int chn_area = this->len_clip * image_area;
    for (int i = 0; i < this->len_clip; i++)
    {
        vector<cv::Mat> bgrChannels(3);
        split(video_clips[current_rate][i], bgrChannels);
        memcpy(this->input_tensor.data() + i * image_area, (float *)bgrChannels[0].data, single_chn_size);
        memcpy(this->input_tensor.data() + chn_area + i * image_area, (float *)bgrChannels[1].data, single_chn_size);
        memcpy(this->input_tensor.data() + 2 * chn_area + i * image_area, (float *)bgrChannels[2].data, single_chn_size);

    }
    current_rate += 1;
    if (current_rate >= rate)
    {
        current_rate = 0;
    }

    const int origin_h = input_mat.rows;
    const int origin_w = input_mat.cols;
    // this->preprocess(video_clip);

    std::vector<int64_t> input_img_shape = {1, 3, this->len_clip, this->inpHeight, this->inpWidth};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_tensor.data(), this->input_tensor.size(), input_img_shape.data(), input_img_shape.size());

    Ort::RunOptions runOptions;
    this->start_time = std::chrono::system_clock::now();
    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());
    this->end_time = std::chrono::system_clock::now();
    diff = this->end_time - this->start_time;
    // cout << "向前推理时间：" << diff.count() << endl;
    diffs.emplace_back(diff.count());
    const float *conf_preds0 = ort_outputs[0].GetTensorMutableData<float>();
    const float *conf_preds1 = ort_outputs[1].GetTensorMutableData<float>();
    const float *conf_preds2 = ort_outputs[2].GetTensorMutableData<float>();
    const float *cls_preds0 = ort_outputs[3].GetTensorMutableData<float>();
    const float *cls_preds1 = ort_outputs[4].GetTensorMutableData<float>();
    const float *cls_preds2 = ort_outputs[5].GetTensorMutableData<float>();
    const float *reg_preds0 = ort_outputs[6].GetTensorMutableData<float>();
    const float *reg_preds1 = ort_outputs[7].GetTensorMutableData<float>();
    const float *reg_preds2 = ort_outputs[8].GetTensorMutableData<float>();

    this->generate_proposal_one_hot(this->strides[0], conf_preds0, cls_preds0, reg_preds0, boxes, det_conf, cls_conf);
    this->generate_proposal_one_hot(this->strides[1], conf_preds1, cls_preds1, reg_preds1, boxes, det_conf, cls_conf);
    this->generate_proposal_one_hot(this->strides[2], conf_preds2, cls_preds2, reg_preds2, boxes, det_conf, cls_conf);

    vector<int> keep_inds = multiclass_nms_class_agnostic(boxes, det_conf, this->nms_thresh);

    const int max_hw = max(this->inpHeight, this->inpWidth);
    const float ratio_h = float(origin_h) / max_hw;
    const float ratio_w = float(origin_w) / max_hw;
    for (int i = 0; i < keep_inds.size(); i++)
    {
        const int ind = keep_inds[i];
        boxes[ind].xmin = int((float)boxes[ind].xmin * ratio_w);
        boxes[ind].ymin = int((float)boxes[ind].ymin * ratio_h);
        boxes[ind].xmax = int((float)boxes[ind].xmax * ratio_w);
        boxes[ind].ymax = int((float)boxes[ind].ymax * ratio_h);
    }
    return keep_inds;
}

std::map<unsigned long, vector<float>> TAD::detect_multi_hot(const std::map<size_t, Mat>& video_mat_with_track_id)
{
    std::map<size_t, Mat> preprocessed_track_mats;
    for(auto &_vm : video_mat_with_track_id)
    {
        preprocessed_track_mats[_vm.first] = preprocess(_vm.second);
    }

    for(auto &_vm : video_mat_with_track_id)
    {
        if(multi_video_clips.count(_vm.first) == 0)
        {

            for (int i = 0; i < rate; i ++)
            {
                vector<Mat> _temp;
                for (size_t z = 0; z < len_clip ; z ++)
                {
                    _temp.emplace_back(preprocessed_track_mats[_vm.first]);
                }
                multi_video_clips[_vm.first].emplace_back(_temp);
            }
        }else
        {
            multi_video_clips[_vm.first][current_rate].erase(multi_video_clips[_vm.first][current_rate].begin());
            multi_video_clips[_vm.first][current_rate].emplace_back(preprocessed_track_mats[_vm.first]);
        }

    }


    batch_size = video_mat_with_track_id.size();
    const int image_area = this->inpHeight * this->inpWidth;
    input_tensor.resize(batch_size * 3 * this->len_clip * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    const int chn_area = this->len_clip * image_area;
    const int batch_area = chn_area * 3;
    int _b = -1;
    for(auto &_vm : video_mat_with_track_id)
    {
        _b += 1;
        for (int i = 0; i < this->len_clip; i++)
        {
            vector<cv::Mat> bgrChannels(3);
            split(multi_video_clips[_vm.first][current_rate][i], bgrChannels);
            memcpy(this->input_tensor.data() + batch_area * _b + 0 * chn_area + i * image_area, (float *)bgrChannels[0].data, single_chn_size);
            memcpy(this->input_tensor.data() + batch_area * _b + 1 * chn_area + i * image_area, (float *)bgrChannels[1].data, single_chn_size);
            memcpy(this->input_tensor.data() + batch_area * _b + 2 * chn_area + i * image_area, (float *)bgrChannels[2].data, single_chn_size);

        }
    }

    current_rate += 1;
    if (current_rate >= rate)
    {
        current_rate = 0;
    }
    // this->preprocess(video_clip);

    std::vector<int64_t> input_img_shape = {batch_size, 3, this->len_clip, this->inpHeight, this->inpWidth};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_tensor.data(), this->input_tensor.size(), input_img_shape.data(), input_img_shape.size());

    Ort::RunOptions runOptions;
    this->start_time = std::chrono::system_clock::now();
    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());
    this->end_time = std::chrono::system_clock::now();
    diff = this->end_time - this->start_time;
    cout << "TAD 向前推理时间：" << diff.count() << endl;
    diffs.emplace_back(diff.count());
    const float *_conf_preds = ort_outputs[0].GetTensorMutableData<float>();

    auto* conf_preds = const_cast<float*>(_conf_preds);
    softmax(conf_preds, batch_size, num_class);


    _b = -1;
    std::map<size_t, vector<float>> result;
    for(auto &_vm : video_mat_with_track_id)
    {
        _b += 1;
        vector<float> _temp;
        result[_vm.first] = _temp;
        for(size_t _l = 0; _l < num_class; _l ++)
        {
            result[_vm.first].emplace_back(conf_preds[_b * num_class + _l]);
        }
    }

    return result;


}

Mat TAD::vis_one_hot(Mat frame, const vector<Bbox> boxes, const vector<float> det_conf, const vector<vector<float>> cls_conf, const vector<int> keep_inds, const float vis_thresh)
{
    Mat dstimg = frame.clone();
    for (int i = 0; i < keep_inds.size(); i++)
    {
        const int ind = keep_inds[i];
        vector<int> indices;
        vector<float> scores;
        for (int j = 0; j < cls_conf[ind].size(); j++)
        {
            const float det_cls_conf = sqrt(det_conf[ind] * cls_conf[ind][j]);
            if (det_cls_conf > vis_thresh)
            {
                scores.emplace_back(det_cls_conf);
                indices.emplace_back(j);
            }
        }

        if (scores.size() > 0)
        {
            int xmin = boxes[ind].xmin;
            int ymin = boxes[ind].ymin;
            int xmax = boxes[ind].xmax;
            int ymax = boxes[ind].ymax;
            rectangle(dstimg, Point(xmin, ymin), Point(xmax, ymax), Scalar(0, 0, 255), 2);

            // Mat blk = Mat::zeros(frame.rows, frame.cols, CV_8UC3);
            for (int j = 0; j < indices.size(); j++)
            {
                string text = format("[%.2f] ", scores[j]) + labels[indices[j]];
                int baseline = 0;
                Size text_size = getTextSize(text, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseline);
                const int coord_x = xmin + 3;
                const int coord_y = ymin + 14 + 20 * j;
                rectangle(dstimg, Point(coord_x - 1, coord_y - 12), Point(coord_x + text_size.width + 1, coord_y + text_size.height - 4), Scalar(0, 0, 255), -1);
                putText(dstimg, text, Point(coord_x, coord_y), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255), 1);
            }
            // addWeighted(dstimg, 1.0, blk, 0.5, 1, dstimg);
        }
    }
    return dstimg;
}
