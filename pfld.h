//
// Created by wangke on 2023/5/15.
//

#ifndef PFLD_POSE_H
#define PFLD_POSE_H

#include <opencv2/core/core.hpp>

#include <opencv2/highgui.hpp>
#include <net.h>

#include <stdio.h>

#include <vector>

struct Object_pose_pfld
{
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> kps;
};


class pfld_pose
{
public:
    pfld_pose();
    int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    int detect(const cv::Mat& rgb, ncnn::Mat& result_mat, float prob_threshold = 0.25f, float nms_threshold = 0.65f);

    int draw(cv::Mat& rgb, ncnn::Mat& result_mat);



private:
    ncnn::Net yolo;
    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif 
