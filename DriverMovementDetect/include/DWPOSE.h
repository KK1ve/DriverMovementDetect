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
        explicit DWPOSE(const string& modelpath, int _originHeight, int _originWidth, int use_int8 = 0, float nms_thresh_ = 0.5, float conf_thresh_ = 0.6);
        void detect(const std::vector<Mat>& track_imgs, vector<ObjectPose> &boxes);
        Mat vis(const Mat& frame, vector<ObjectPose> &boxes);

        vector<float> diffs;

    private:
        std::chrono::system_clock::time_point start_time;
        std::chrono::system_clock::time_point end_time;
        std::chrono::duration<float> diff;

        vector<float> input_tensor;
        Mat preprocess(const Mat& video_mat);
        int inpBatchSize;
        int inpWidth;
        int inpHeight;
        int originHeight;
        int originWidth;
        float nms_thresh;
        float conf_thresh;

        void generate_proposal(const vector<float>& keypoints, vector<ObjectPose> &boxes);

        vector<char *> input_names;
        vector<char *> output_names;
        vector<vector<int64_t>> input_node_dims;  // >=1 outputs
        vector<vector<int64_t>> output_node_dims; // >=1 outputs

        std::shared_ptr<BMNNContext> m_bmContext;
        std::shared_ptr<BMNNNetwork> m_bmNetwork;
        std::shared_ptr<BMNNTensor>  m_input_tensor;
        bm_tensor_t bm_input_tensor;

        const vector<float> means = {123.675, 116.28, 103.53};
        const vector<float> stds = {1.0/58.395, 1.0/57.12, 1.0/57.375};
        int num_keypoint;


};



#endif //DWPOSE_H
