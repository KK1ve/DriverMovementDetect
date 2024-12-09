//
// Created by 171153 on 2024/10/17.
//

#ifndef DWPOSE_H
#define DWPOSE_H

#include <opencv2/highgui.hpp>
#include <cmath>
#include "bmnn_utils.h"
#include <string>
#include <utils.h>
#include <vector>
using namespace std;
using namespace cv;


class DWPOSE {
    public:
        explicit DWPOSE(const string& modelpath, int _originHeight, int _originWidth);
        void detect(const std::vector<Mat>& track_imgs, vector<ObjectPose> &boxes);
        Mat vis(const Mat& frame, vector<ObjectPose> &boxes);

        CommonResultPose pre_process(CommonResultPose& input);
        CommonResultPose detect(CommonResultPose& input);
        CommonResultPose post_process(CommonResultPose& input);

    private:

        Mat preprocess(const Mat& video_mat);
        int inpBatchSize;
        int inpWidth;
        int inpHeight;
        int originHeight;
        int originWidth;

        void generate_proposal(const vector<float>& keypoints, vector<ObjectPose> &boxes);

        std::shared_ptr<BMNNContext> mBMContext;
        std::shared_ptr<BMNNNetwork> mBMNetwork;
        bm_tensor_t bmInputTensor;

        const vector<float> means = {123.675, 116.28, 103.53};
        const vector<float> stds = {1.0/58.395, 1.0/57.12, 1.0/57.375};
        int keypointsCount;


};



#endif //DWPOSE_H
