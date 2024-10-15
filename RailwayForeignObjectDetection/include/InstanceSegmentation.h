//
// Created by 171153 on 2024/8/12.
//

#ifndef INSTANCESEGMENTATION_H
#define INSTANCESEGMENTATION_H



#include <cmath>
#include <string>
#include <vector>
#include "bmruntime_interface.h"
#include "bmcv_api_ext.h"
#include "bmlib_runtime.h"
#include "opencv2/opencv.hpp"
#include "bm_wrapper.h"
#include "bmnn_utils.h"

using namespace std;
using namespace cv;


struct ObjectSeg
{
    Mat region;
    int label;
    string label_name;
};


class IS
{
    public:
        explicit IS(const string& modelpath, float nms_thresh_ = 0.5, float conf_thresh_ = 0.6);
        void detect(const Mat& video_mat, vector<ObjectSeg> &boxes);
        Mat vis(const Mat& frame, vector<ObjectSeg> &boxes);

        vector<float> diffs;

    private:
        std::chrono::system_clock::time_point start_time;
        std::chrono::system_clock::time_point end_time;
        std::chrono::duration<float> diff;

        vector<float> input_tensor;
        Mat preprocess(const Mat& video_mat);
        int inpWidth;
        int inpHeight;
        float nms_thresh;
        float conf_thresh;



        void generate_proposal(const float *pred, vector<ObjectSeg> &boxes);

        vector<char *> input_names;
        vector<char *> output_names;
        vector<vector<int64_t>> input_node_dims;  // >=1 outputs
        vector<vector<int64_t>> output_node_dims; // >=1 outputs

        std::shared_ptr<BMNNContext> m_bmContext;
        std::shared_ptr<BMNNNetwork> m_bmNetwork;
        std::shared_ptr<BMNNTensor>  m_input_tensor;
        std::shared_ptr<BMNNTensor>  m_output_tensor;
        bm_tensor_t bm_input_tensor;

        const char* labels[3] = {"rail-raised", "rail-track", "unidentified"};
        const int num_class = 3;
        const vector<vector<int>> mat_colos = {
            {0, 0, 255},
            {255, 255, 102},
            {0, 0, 0}
        };

};




#endif //INSTANCESEGMENTATION_H
