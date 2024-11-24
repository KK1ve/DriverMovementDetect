//
// Created by 171153 on 2024/8/6.
//

#include "../include/YOLOV3.h"
#include "../include/utils.h"
#include <numeric>
#include <algorithm>
#include <memory>
#include <iostream>
#include <opencv2/imgproc.hpp>
static bool cmp_score(const ObjectPose& box1, const ObjectPose& box2) {
    // if (box1.prob == box2.prob)
    // {
    //     if(box1.rect.x == box2.rect.x)
    //     {
    //         return box1.rect.y > box2.rect.y;
    //     }
    //     return box1.rect.x > box2.rect.x;
    // }
    return box1.prob > box2.prob;
}

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

std::vector<int> multiclass_nms_class_agnostic(std::vector<ObjectPose> &boxes, const float nms_thresh)
{
    std::sort(boxes.begin(), boxes.end(), cmp_score);
    const int num_box = boxes.size();
    std::vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        vector<float> ious(boxes.size());

        #pragma omp parallel for
        for (int j = i + 1; j < boxes.size(); ++j)
        {
            ious[j] = GetIoU(boxes[i].rect, boxes[j].rect);
        }

        for (int j = i + 1; j < num_box; ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            if (ious[j] > nms_thresh)
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

YOWOV3::YOWOV3(const string& modelpath, const float nms_thresh_, const float conf_thresh_, const int _rate)
{
    // creat handle
    BMNNHandlePtr handle = make_shared<BMNNHandle>(0);
    bm_handle_t h = handle->handle();

    m_bmContext = make_shared<BMNNContext>(handle, modelpath.c_str());


    // 1. get network
    m_bmNetwork = m_bmContext->network(0);

    m_input_tensor = m_bmNetwork->inputTensor(0);
    m_output_tensor = m_bmNetwork->outputTensor(0);

    bmrt_tensor(&bm_input_tensor,
                m_bmContext->bmrt(),
                m_input_tensor->get_dtype(),
                *m_input_tensor->get_shape());
    m_input_tensor->set_device_mem(&bm_input_tensor.device_mem);

    this->len_clip = m_input_tensor->get_shape()->dims[2];
    this->inpHeight = m_input_tensor->get_shape()->dims[3];
    this->inpWidth = m_input_tensor->get_shape()->dims[4];
    this->conf_thresh = conf_thresh_;
    this->nms_thresh = nms_thresh_;
    rate = _rate;

    for (int i = 0; i < rate; i++)
    {
        vector<Mat> _temp;
        video_clips.emplace_back(_temp);
    }

}

/*void YOWOV3::preprocess(vector<Mat> video_clip)
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

Mat YOWOV3::preprocess(const Mat& video_mat)
{
    Mat resizeimg;
    resize(video_mat, resizeimg, cv::Size(this->inpHeight, this->inpWidth));
    resizeimg.convertTo(resizeimg, CV_32SC3);
    cv::Scalar mean(means[0], means[1], means[2]);
    cv::Mat mean_mat(resizeimg.rows, resizeimg.cols, CV_32SC3, mean);
    cv::subtract(resizeimg, mean_mat, resizeimg);
    return resizeimg;

}

void YOWOV3::generate_proposal_one_hot(const float *pred, vector<ObjectPose> &boxes, int origin_w, int origin_h, int precessed_w, int precessed_h)
{
    int batch_size = m_output_tensor->get_shape()->dims[0];
    int K = m_output_tensor->get_shape()->dims[1];
    int M = m_output_tensor->get_shape()->dims[2];

    vector<ObjectPose> generated_result;
    for (int n = 0; n < batch_size; n++)
    {
        for (int i = 0; i < M; i ++)
        {
            float max_score = -FLT_MAX;
            vector<float> action_prob;
            for (int k = 0; k < (K - 4); k ++)
            {
                if (pred[n * K * M + (k + 4) * M + i] > max_score) max_score = pred[n * K * M + (k + 4) * M + i];
                action_prob.emplace_back(pred[n * K * M + (k + 4) * M + i]);
            }
            if (max_score < conf_thresh) continue;
            ObjectPose op;
            op.label = 0;
            cv::Rect t(pred[n * K * M + 0 * M + i] - (pred[n * K * M + 2 * M + i] / 2),pred[n * K * M + 1 * M + i] - (pred[n * K * M + 3 * M + i] / 2),
                pred[n * K * M + 2 * M + i],pred[n * K * M + 3 * M + i]);
            op.rect = t;
            op.prob = max_score;
            op.action_prob = action_prob;
            generated_result.emplace_back(op);
        }
    }
    auto keepidx = multiclass_nms_class_agnostic(generated_result, nms_thresh);
    boxes.clear();
    for (auto i : keepidx)
    {
        boxes.emplace_back(generated_result[i]);
    }

    float r = std::min(float(precessed_w) / origin_w, float(precessed_h) / origin_h);
    int inside_w = round(origin_w * r);
    int inside_h = round(precessed_w * r);

    int padd_w = (precessed_w - inside_w) / 2;
    int padd_h = (precessed_h - inside_h) / 2;

    for (auto& box: boxes)
    {
        cv::Rect _rect((box.rect.x - padd_w) / r, (box.rect.y - padd_h) / r, (box.rect.width - padd_w * 2) / r, (box.rect.height - padd_h * 2) / r);
        box.rect = _rect;
    }
}

void YOWOV3::clear_clips_cache()
{
    multi_video_clips.clear();
}

void YOWOV3::detect_one_hot(const Mat& input_mat, vector<ObjectPose> &boxes)
{
    this->origin_h = input_mat.rows;
    this->origin_w = input_mat.cols;
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
    input_tensor.resize(1 * this->len_clip * 3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    const int chn_area = this->len_clip * image_area;
    for (int i = 0; i < this->len_clip; i++)
    {
        vector<cv::Mat> bgrChannels(3);
        split(video_clips[current_rate][i], bgrChannels);
        for (int c = 0; c < 3; c ++)
        {
            bgrChannels[c].convertTo(bgrChannels[c],CV_32SC3, this->stds[c]);
        }
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
    cout << "向前推理时间：" << diff.count() << endl;
    diffs.emplace_back(diff.count());

    auto temp_result = m_bmNetwork->outputTensor(0);
    auto result = temp_result->get_cpu_data();

    generate_proposal_one_hot(result, boxes, input_mat.cols, input_mat.rows, preprocessed_mat.cols,
                              preprocessed_mat.rows);

}

Mat YOWOV3::vis_one_hot(const Mat& frame, const vector<ObjectPose>& boxes, const bool show_action, const float action_thresh, const float keypoint_thresh)
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

    const int num_points = 18;

    cv::Mat res = frame.clone();
    for (auto& obj : boxes)
    {
        cv::rectangle(res, obj.rect, { 0, 0, 255 }, 2);
        imwrite("TEST.jpg", res);
        string text = format("%s %.1f%% \n", labels[obj.label], obj.prob * 100);
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
            text += format("track_id: %u \n", obj.track_id);
            if(show_action)
            {
                for (int i = 0; i < obj.action_prob.size(); i ++)
                {
                    if(obj.action_prob[i] > action_thresh)
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
        for (int k = 0; k < num_points; k++)
        {
            int kps_x= std::round(kps[k * 3]);
            int kps_y=std::round(kps[k * 3 + 1]);
            float kps_s = kps[k * 3 + 2];


            if (kps_s > 0.5f)
            {
                cv::Scalar kps_color = cv::Scalar(KPS_COLORS[(int)k % 17][0], KPS_COLORS[(int)k % 17][1], KPS_COLORS[(int)k % 17][2]);
                cv::circle(res, { kps_x, kps_y }, 5, kps_color, -1);
            }

        }
        imwrite("TEST.jpg", res);
    }
    return res;
}
