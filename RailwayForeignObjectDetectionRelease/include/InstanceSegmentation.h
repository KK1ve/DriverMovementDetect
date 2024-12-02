//
// Created by 171153 on 2024/8/12.
//

#ifndef INSTANCESEGMENTATION_H
#define INSTANCESEGMENTATION_H

#include <string>
#include <vector>
#include "opencv2/opencv.hpp"
#include "bmnn_utils.h"
#include <tbb/concurrent_vector.h>

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
        explicit IS(const string& modelpath, int use_int8 = 0, float nms_thresh_ = 0.5, float conf_thresh_ = 0.6);
        // tuple [frame_id, last_step_result, origin_mat]

        std::tuple<unsigned long, std::vector<float>, Mat> pre_process(std::tuple<unsigned long, Mat>& mat);
        std::tuple<unsigned long, std::shared_ptr<BMNNTensor>, Mat> detect(std::tuple<unsigned long, std::vector<float>, Mat>& input_vector);
        std::tuple<unsigned long, vector<ObjectSeg>, Mat> post_process(std::tuple<unsigned long, std::shared_ptr<BMNNTensor>, Mat>& pred);
        std::tuple<unsigned long, Mat, Mat> vis(std::tuple<unsigned long, vector<ObjectSeg>, Mat>& boxes);


    private:
        int batchSize;
        int inpWidth;
        int inpHeight;
        float nms_thresh;
        float conf_thresh;

        void generateProposal(const float *pred, vector<ObjectSeg> &boxes);


        std::shared_ptr<BMNNContext> mBMContext;
        std::shared_ptr<BMNNNetwork> mBMNetwork;
        bm_tensor_t bmInputTensor;

        const char* labels[3] = {"rail-raised", "rail-track", "unidentified"};
        const int numClass = 3;
        const vector<vector<int>> matColos = {
            {0, 0, 255},
            {255, 255, 102},
            {0, 0, 0}
        };

};




#endif //INSTANCESEGMENTATION_H
