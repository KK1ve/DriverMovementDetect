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


TAD::TAD(const string& modelpath,const int _origin_h, const int _origin_w, const float nms_thresh_, const float conf_thresh_)
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
    this->len_clip = m_input_tensor->get_shape()->dims[2];
    this->inpHeight = m_input_tensor->get_shape()->dims[3];
    this->inpWidth = m_input_tensor->get_shape()->dims[4];
    this->confThresh = conf_thresh_;
    this->nmsThresh = nms_thresh_;
    this->originHeight = _origin_h;
    this->originWidth = _origin_w;


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
    auto resizeimg = letterbox(video_mat, this->inpHeight, this->inpWidth);
    resizeimg.convertTo(resizeimg, CV_32FC3);
    return resizeimg;

}

void TAD::generate_proposal_one_hot(const int stride, const float *conf_pred, const float *cls_pred, const float *reg_pred, vector<ObjectPose> &boxes)
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

    for (int ind : topk_inds)
    {
        if (conf_pred_i[ind] > this->conf_thresh)
        {
            int row = 0, col = 0;
            ind2sub(ind, feat_w, feat_h, row, col);

            float cx = (col + 0.5f + reg_pred[ind * 4]) * stride;
            float cy = (row + 0.5f + reg_pred[ind * 4 + 1]) * stride;
            float w = exp(reg_pred[ind * 4 + 2]) * stride;
            float h = exp(reg_pred[ind * 4 + 3]) * stride;
            ObjectPose op;
            op.label = 0;
            cv::Rect _rect(int(cx - 0.5 * w), int(cy - 0.5 * h), int(cx + 0.5 * w), int(cy + 0.5 * h));
            op.rect = _rect;
            op.prob = conf_pred_i[ind];

            vector<float> cls_conf_i(this->num_class);
            for (int j = 0; j < this->num_class; j++)
            {
                cls_conf_i[j] = sigmoid(cls_pred[ind * this->num_class + j]);
            }
            op.action_prob = cls_conf_i;
            boxes.emplace_back(op);
        }
    }
}

void TAD::clear_clips_cache()
{
    multi_video_clips.clear();
}

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
    tbb::parallel_for(0, len_clip, 1, [&](int i) {
        vector<cv::Mat> bgrChannels(3);
        split(mats[i], bgrChannels);
        tbb::parallel_for(0, 3, 1, [&](int j)
        {
            memcpy(input_tensor.data() + j * chn_area + i * image_area, (float *)bgrChannels[j].data, single_chn_size);
        });
    });
    return input_tensor;

}

CommonResultPose TAD::detect(CommonResultPose& input)
{
    bm_memcpy_s2d(mBMContext->handle(), bmInputTensor.device_mem, input.float_vector.data());
    mBMNetwork->forward();

}

void TAD::detect_one_hot(const Mat& input_mat, vector<ObjectPose> &boxes)
{
    this->origin_h = input_mat.rows;
    this->origin_w = input_mat.cols;
    Mat preprocessed_mat = preprocess(input_mat);
    if (video_clips.size() <= current_rate && video_clips.size() < rate)
    {
        vector<Mat> _;
        video_clips.emplace_back(_);
    }
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
    input_tensor.resize(1 * this->len_clip * 3 * image_area);
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

    bm_memcpy_s2d(m_bmContext->handle(), bm_input_tensor.device_mem, &input_tensor[0]);

    this->start_time = std::chrono::system_clock::now();
    m_bmNetwork->forward();
    this->end_time = std::chrono::system_clock::now();
    diff = this->end_time - this->start_time;
    cout << "TAD向前推理时间：" << diff.count() << endl;
    diffs.emplace_back(diff.count());

    auto conf_pred_0 = m_bmNetwork->outputTensor(0);
    auto conf_pred_1 = m_bmNetwork->outputTensor(1);
    auto conf_pred_2 = m_bmNetwork->outputTensor(2);
    auto cls_pred_0 = m_bmNetwork->outputTensor(3);
    auto cls_pred_1 = m_bmNetwork->outputTensor(4);
    auto cls_pred_2 = m_bmNetwork->outputTensor(5);
    auto reg_pred_0 = m_bmNetwork->outputTensor(6);
    auto reg_pred_1 = m_bmNetwork->outputTensor(7);
    auto reg_pred_2 = m_bmNetwork->outputTensor(8);
    vector<ObjectPose> _temp_boxes;
    this->generate_proposal_one_hot(this->strides[0], conf_pred_0->get_cpu_data(),
        cls_pred_0->get_cpu_data(), reg_pred_0->get_cpu_data(), _temp_boxes);
    this->generate_proposal_one_hot(this->strides[1], conf_pred_1->get_cpu_data(),
        cls_pred_1->get_cpu_data(), reg_pred_1->get_cpu_data(), _temp_boxes);
    this->generate_proposal_one_hot(this->strides[2], conf_pred_2->get_cpu_data(),
        cls_pred_2->get_cpu_data(), reg_pred_2->get_cpu_data(), _temp_boxes);

    vector<int> keep_inds = multiclass_nms_class_agnostic(_temp_boxes, this->nms_thresh);


    for (int ind :keep_inds)
    {
        boxes.emplace_back(_temp_boxes[ind]);
    }
    const float heightScale = float(inpHeight) / float(origin_h), widthScale = float(inpWidth) / float(origin_w);
    const float scale = min(heightScale, widthScale);
    const int padd_w = round((float(inpWidth) - float(origin_w) * scale) / 2.0f);
    const int padd_h = round((float(inpHeight) - float(origin_h) * scale) / 2.0f);

    for(auto& box: boxes)
    {
        box.rect.width = int(float(box.rect.width - box.rect.x) / scale) + padd_w;
        box.rect.height = int(float(box.rect.height - box.rect.y) / scale) + padd_h;
        box.rect.x = int(float(box.rect.x - padd_w) / scale);
        box.rect.y = int(float(box.rect.y - padd_h) / scale);


        box.rect.x = box.rect.x > origin_w ? origin_w : box.rect.x < 0 ? 0 : box.rect.x;
        box.rect.y = box.rect.y > origin_h ? origin_h : box.rect.y < 0 ? 0 : box.rect.y;
        box.rect.width = box.rect.width > origin_w - box.rect.x ? origin_w - box.rect.x : box.rect.width;
        box.rect.height = box.rect.height > origin_h - box.rect.y ? origin_h - box.rect.y : box.rect.height;
    }
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
    input_tensor.resize(3 * this->len_clip * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    const int chn_area = this->len_clip * image_area;

    vector<float> pre_result(batch_size * num_class);
    for(auto &_vm : video_mat_with_track_id)
    {
        input_tensor.clear();
        for (int i = 0; i < this->len_clip; i++)
        {
            vector<cv::Mat> bgrChannels(3);
            split(multi_video_clips[_vm.first][current_rate][i], bgrChannels);
            memcpy(this->input_tensor.data() + 0 * chn_area + i * image_area, (float *)bgrChannels[0].data, single_chn_size);
            memcpy(this->input_tensor.data() + 1 * chn_area + i * image_area, (float *)bgrChannels[1].data, single_chn_size);
            memcpy(this->input_tensor.data() + 2 * chn_area + i * image_area, (float *)bgrChannels[2].data, single_chn_size);
        }
        bm_memcpy_s2d(m_bmContext->handle(), bm_input_tensor.device_mem, &input_tensor[0]);

        this->start_time = std::chrono::system_clock::now();
        m_bmNetwork->forward();
        this->end_time = std::chrono::system_clock::now();
        diff = this->end_time - this->start_time;
        cout << "TAD 向前推理时间：" << diff.count() << endl;
        diffs.emplace_back(diff.count());
        std::shared_ptr<BMNNTensor> outputTensor = m_bmNetwork->outputTensor(0);
        auto conf_preds = outputTensor->get_cpu_data();

        pre_result.insert(pre_result.end(), conf_preds, conf_preds + num_class);
    }

    current_rate += 1;
    if (current_rate >= rate)
    {
        current_rate = 0;
    }
    // this->preprocess(video_clip);



    softmax(&pre_result[0], batch_size, num_class);


    int _b = -1;
    std::map<size_t, vector<float>> result;
    for(auto &_vm : video_mat_with_track_id)
    {
        _b += 1;
        vector<float> _temp;
        result[_vm.first] = _temp;
        for(size_t _l = 0; _l < num_class; _l ++)
        {
            result[_vm.first].emplace_back(pre_result[_b * num_class + _l]);
        }
    }
    pre_result.clear();

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

Mat TAD::vis(const Mat& frame, const vector<ObjectPose>& boxes, const bool show_action, const float action_thresh, const float keypoint_thresh)
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


    cv::Mat res = frame.clone();

    for (auto& obj : boxes)
    {
        cv::rectangle(res, obj.rect, { 0, 0, 255 }, 2);
        string text = format("%s %.1f%% \n", "person", obj.prob * 100);
        // if (obj.label != 0) {
        if (false) {
            text += "infos: \n";
            std::vector<int> right_eyes = { 0, 2, 4, 6, 8, 10};
            std::vector<int> left_eyes = { 1, 3, 5, 7, 9, 11};
            std::vector<int> mouth = { 12, 14, 15, 13, 17, 16};

            float EAR = (euclidean_distance(obj.kps[right_eyes[1] * 3 + 0],
                obj.kps[right_eyes[5] * 3 + 0], obj.kps[right_eyes[1] * 3 + 1],
                obj.kps[right_eyes[5] * 3 + 1]) + euclidean_distance(obj.kps[right_eyes[2] * 3 + 0],
                    obj.kps[right_eyes[4] * 3 + 0], obj.kps[right_eyes[2] * 3 + 1],
                    obj.kps[right_eyes[4] * 3 + 1])) / (2 * euclidean_distance(obj.kps[right_eyes[0] * 3 + 0],
                        obj.kps[right_eyes[3] * 3 + 0], obj.kps[right_eyes[0] * 3 + 1],
                        obj.kps[right_eyes[3] * 3 + 1]));
            text += format("RIGHT: %.2f\n", EAR);


            EAR = (euclidean_distance(obj.kps[left_eyes[1] * 3 + 0],
                obj.kps[left_eyes[5] * 3 + 0], obj.kps[left_eyes[1] * 3 + 1],
                obj.kps[left_eyes[5] * 3 + 1]) + euclidean_distance(obj.kps[left_eyes[2] * 3 + 0],
                    obj.kps[left_eyes[4] * 3 + 0], obj.kps[left_eyes[2] * 3 + 1],
                    obj.kps[left_eyes[4] * 3 + 1])) / (2 * euclidean_distance(obj.kps[left_eyes[0] * 3 + 0],
                        obj.kps[left_eyes[3] * 3 + 0], obj.kps[left_eyes[0] * 3 + 1],
                        obj.kps[left_eyes[3] * 3 + 1]));
            text += format("LEFT: %.2f\n", EAR);


            EAR = (euclidean_distance(obj.kps[mouth[1] * 3 + 0],
                obj.kps[mouth[5] * 3 + 0], obj.kps[mouth[1] * 3 + 1],
                obj.kps[mouth[5] * 3 + 1]) + euclidean_distance(obj.kps[mouth[2] * 3 + 0],
                    obj.kps[mouth[4] * 3 + 0], obj.kps[mouth[2] * 3 + 1],
                    obj.kps[mouth[4] * 3 + 1])) / (2 * euclidean_distance(obj.kps[mouth[0] * 3 + 0],
                        obj.kps[mouth[3] * 3 + 0], obj.kps[mouth[0] * 3 + 1],
                        obj.kps[mouth[3] * 3 + 1]));
            text += format("MOUTH: %.2f", EAR);

            // cv::Size label_size = cv::getTextSize(face_text, cv::FONT_HERSHEY_SIMPLEX,
            //     0.4, 1, &baseLine);
            //
            // cv::putText(res, face_text, cv::Point(x, y - label_size.height),
            //     cv::FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 }, 1);

        }else
        {
            text += format("track id: %u \n", obj.track_id);
            if(show_action)
            {
                for (int i = 0; i < obj.action_prob.size(); i ++)
                {
                    if(obj.action_prob[i] > action_thresh && count(keep_index.begin(), keep_index.end(), i))
                    {
                        text += format("%s:%.1f%% \n",labels[i], obj.action_prob[i] * 100);
                    }
                }
            }
        }

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

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


        std::vector<float> kps = obj.kps;
        if (!kps.empty())
        {
            for (int k = 0; k < 133; k++)
            {
                int kps_x= std::round(kps[k * 3]);
                int kps_y=std::round(kps[k * 3 + 1]);
                float kps_s = kps[k * 3 + 2];
                if (kps_s > keypoint_thresh)
                {
                    cv::Scalar kps_color = cv::Scalar(KPS_COLORS[0][0], KPS_COLORS[0][1], KPS_COLORS[0][2]);
                    cv::circle(res, { kps_x, kps_y }, 0, kps_color, 5);
                }

            }
        }

    }
    return res;
}


std::tuple<unsigned long, std::vector<float>, Mat> TAD::pre_process(std::tuple<unsigned long, Mat>& mat)
{

}
