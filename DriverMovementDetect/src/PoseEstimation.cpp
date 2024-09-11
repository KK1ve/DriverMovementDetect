//
// Created by 171153 on 2024/8/6.
//
#include "../include/PoseEstimation.h"
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <fstream>

static bool cmp_score(const ObjectPose& box1, const ObjectPose& box2) {
    return box1.prob > box2.prob;
}

static float get_iou_value(ObjectPose object_pose1, ObjectPose object_pose2)
{
    int xx1, yy1, xx2, yy2;

    xx1 = std::max(object_pose1.rect.x, object_pose2.rect.x);
    yy1 = std::max(object_pose1.rect.y, object_pose2.rect.y);
    xx2 = std::min(object_pose1.rect.x + object_pose1.rect.width - 1, object_pose2.rect.x + object_pose2.rect.width - 1);
    yy2 = std::min(object_pose1.rect.y + object_pose1.rect.height - 1, object_pose2.rect.y + object_pose2.rect.height - 1);

    int insection_width, insection_height;
    insection_width = std::max(0, xx2 - xx1 + 1);
    insection_height = std::max(0, yy2 - yy1 + 1);

    float insection_area, union_area, iou;
    insection_area = float(insection_width) * insection_height;
    union_area = float(object_pose1.rect.width * object_pose1.rect.height + object_pose2.rect.width * object_pose2.rect.height - insection_area);
    iou = insection_area / union_area;
    return iou;
}

static void my_nms_boxes(const std::vector<ObjectPose>& boxes, float nmsThreshold, std::vector<int>& indices)
{
    std::vector<bool> isSuppressed(boxes.size(), false);
    for (int i = 0; i < boxes.size(); ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < boxes.size(); ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            float ovr = get_iou_value(boxes[i], boxes[j]);
            if (ovr > nmsThreshold)
            {
                isSuppressed[j] = true;
            }
        }
    }

    for (int i = 0; i < isSuppressed.size(); i++)
    {
        if (!isSuppressed[i])
        {
            indices.emplace_back(i);
        }
    }
}

static float intersection_area(const ObjectPose& a, const ObjectPose& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}



PE::PE(const string& modelpath, const float nms_thresh_, const float conf_thresh_)
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

    grid_c = output_node_dims[0][0];
    grid_w = output_node_dims[0][1];
    grid_h = output_node_dims[0][2];
}

Mat PE::preprocess(const Mat& video_mat)
{

    Mat resizeimg;
    resize(video_mat, resizeimg, cv::Size(this->inpWidth, this->inpHeight));
    resizeimg.convertTo(resizeimg, CV_32FC3, 1.0 / 255);
    // cvtColor(resizeimg, resizeimg, cv::COLOR_BGR2RGB);
    return resizeimg;
}

void PE::detect_multi_hot(const Mat& video_mat, vector<ObjectPose> &boxes)
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

    const int origin_h = video_mat.rows;
    const int origin_w = video_mat.cols;

    std::vector<int64_t> input_img_shape = {1, 3, this->inpHeight, this->inpWidth};
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_tensor.data(), this->input_tensor.size(), input_img_shape.data(), input_img_shape.size());

    Ort::RunOptions runOptions;
    this->start_time = std::chrono::system_clock::now();
    vector<Value> ort_outputs = this->ort_session->Run(runOptions, this->input_names.data(), &input_tensor_, this->input_names.size(), this->output_names.data(), this->output_names.size());
    this->end_time = std::chrono::system_clock::now();
    diff = this->end_time - this->start_time;
    cout << "PE 向前推理时间：" << diff.count() << endl;
    diffs.emplace_back(diff.count());
    const float *pred = ort_outputs[0].GetTensorMutableData<float>();
    generate_proposal_multi_hot(pred, boxes);
    const int max_hw = max(this->inpHeight, this->inpWidth);
    const float ratio_h = float(origin_h) / max_hw;
    const float ratio_w = float(origin_w) / max_hw;
    for (auto& box: boxes)
    {
        box.rect.x = int((float)box.rect.x * ratio_w);
        box.rect.y = int((float)box.rect.y * ratio_h);
        box.rect.width = int((float)box.rect.width * ratio_w);
        box.rect.height = int((float)box.rect.height * ratio_h);
        for (int i = 0; i < (grid_w - 4 - num_class) / 3; i ++)
        {
            box.kps[3 * i + 0] = int((float)box.kps[3 * i + 0] * ratio_w);
            box.kps[3 * i + 1] = int((float)box.kps[3 * i + 1] * ratio_h);
        }

    }

}

void PE::generate_proposal_multi_hot(const float *pred, vector<ObjectPose> &boxes)
{

    const int kps_num = (grid_w - 4 - num_class) / 3;
    std::vector<std::vector<float>> table(grid_h, std::vector<float>(grid_w));
    /*fstream f("data.csv");
    for (int i = 0 ; i < grid_h; i ++)
    {
        f << i << ",";
    }
    f << endl;

    for (int y = 0 ; y < grid_w; y ++)
    {
        for (int x = 0; x < grid_h; x ++)
        {
            table[x][y] = pred[y * grid_h + x];
            f << pred[y * grid_h + x] << ",";
        }
        f << endl;
    }*/



    vector<vector<ObjectPose>> generated_result(num_class, vector<ObjectPose>(0));
    for (int x = 0 ; x < grid_h; x ++)
    {
        for (int y = 0; y < num_class; y++)
        {
            // table[x][4 + y]
            if (pred[(4 + y) * grid_h + x] >= conf_thresh)
            {
                std::vector<float> kps;
                for (int k = 0; k < kps_num; k++)
                {
                    // table[x][4 + num_classes + k * 3]
                    float kps_x = pred[(4 + num_class + k * 3) * grid_h + x];
                    float kps_y = pred[(4 + num_class + k * 3 + 1) * grid_h + x];
                    float kps_s = pred[(4 + num_class + k * 3 + 2) * grid_h + x];

                    kps.push_back(kps_x);
                    kps.push_back(kps_y);
                    kps.push_back(kps_s);
                }
                ObjectPose temp_object_pose;
                temp_object_pose.kps = kps;
                temp_object_pose.label = y;
                // table[x][4 + y]
                temp_object_pose.prob = pred[(4 + y) * grid_h + x];
                // table[x][0] table[x][2]
                temp_object_pose.rect.x = pred[0 * grid_h + x] - pred[2 * grid_h + x] / 2;
                // table[x][1] table[x][3]
                temp_object_pose.rect.y = pred[1 * grid_h + x] - pred[3 * grid_h + x] / 2;
                // table[x][2]
                temp_object_pose.rect.width = pred[2 * grid_h + x];
                // table[x][3]
                temp_object_pose.rect.height = pred[3 * grid_h + x];
                generated_result[y].emplace_back(temp_object_pose);
                break;
            }
        }
    }
    for (int la = 0; la < num_class; la++) {
        if (generated_result[la].empty()) {
            continue;
        }
        std::vector<int> indices;
        sort(generated_result[la].begin(), generated_result[la].end(), cmp_score);
        my_nms_boxes(
            generated_result[la],
            nms_thresh,
            indices
        );
        for (int i : indices)
        {

            auto& bbox = generated_result[la][i].rect;

            float& score = generated_result[la][i].prob;
            int& label = generated_result[la][i].label;
            auto& kpss = generated_result[la][i].kps;

            /*float x0 = clamp((bbox.x - dw) / ratio_h, 0.f, orin_w);
            float y0 = clamp((bbox.y - dh) / ratio_h, 0.f, orin_h);
            float width = clamp(bbox.width, 0.f, orin_w);
            float height = clamp(bbox.height, 0.f, orin_w);*/


            ObjectPose obj;
            obj.rect.x = bbox.x;
            obj.rect.y = bbox.y;
            obj.rect.width = bbox.width;
            obj.rect.height = bbox.height;
            obj.prob = score;
            obj.label = label;
            obj.kps = kpss;
            for (int n = 0; n < obj.kps.size(); n += 3)
            {
                obj.kps[n] = obj.kps[n];
                obj.kps[n + 1] = obj.kps[n + 1];
            }
            boxes.push_back(obj);
        }

    }
}

float euclidean_distance(float x1, float x2, float y1, float y2) {
    return sqrt((pow(abs(x1 - x2), 2)) + (pow(abs(y1 - y2), 2)));
}



void drawMultilineText(cv::Mat &image, const std::string &text, cv::Point org, int fontFace,
                       double fontScale, cv::Scalar color, int thickness, int lineType, int baseline)
{
    // 按行分割文本
    std::istringstream iss(text);
    std::string line;
    int y = org.y;

    while (std::getline(iss, line)) {
        cv::putText(image, line, cv::Point(org.x, y), fontFace, fontScale, color, thickness, lineType);
        // 计算下一行的 y 坐标，假设行高等于文本的高度加上 baseline
        y += cv::getTextSize(line, fontFace, fontScale, thickness, &baseline).height + baseline;
    }
}

Mat PE::vis_multi_hot(const Mat& frame, const vector<ObjectPose>& boxes, const bool show_action, const float action_thresh)
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
        string text = format("%s %.1f%% \n", labels[obj.label], obj.prob * 100);
        if (obj.label != 0) {
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
                        text += format("%s:%.1f%% \n",action_labels[i], obj.action_prob[i] * 100);
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
    }
    return res;
}



