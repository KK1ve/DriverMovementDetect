//
// Created by wangke on 2023/5/15.
//

#ifndef Yolo_Pose_H
#define Yolo_Pose_H

#include <opencv2/core/core.hpp>

#include <opencv2/highgui.hpp>
#include <net.h>
#include <cmath>

#include <stdio.h>
#include <string>
#include <vector>
#include <string>
struct Object_Pose
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> kps;
    std::string pose_name;
};


class Yolo_Pose
{
public:
    Yolo_Pose();
    int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, std::vector<std::string> class_name, const bool use_gpu = false);

    int detect(const cv::Mat& rgb, std::vector<Object_Pose>& objects, float prob_threshold = 0.70f, float nms_threshold = 0.65f);

    int draw(cv::Mat& rgb, std::vector<Object_Pose>& objects);


private:
    ncnn::Net yolo;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    std::vector<std::string> class_names;
    int num_classes;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif //Yolo_Pose_H
