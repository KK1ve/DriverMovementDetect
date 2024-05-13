#ifndef POSE_CLASSIFICATION_H
#define POSE_CLASSIFICATION_H

#include <opencv2/core/core.hpp>

#include <opencv2/highgui.hpp>
#include <net.h>

#include <stdio.h>
#include <string>
#include <vector>
#include "YoloPose.h"
class Pose_Classification
{
public:
    Pose_Classification();
    int load(const char* modeltype,const int img_width, const int img_height, std::vector<std::string> pose_name, const bool use_gpu = true);

    int detect(std::vector<Object_Pose>& objects);



private:
    std::vector<std::string> pose_name;
    int img_width;
    int img_height;
    ncnn::Net pose_classification;
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;
};

#endif //POSE_CLASSIFICATION_H
