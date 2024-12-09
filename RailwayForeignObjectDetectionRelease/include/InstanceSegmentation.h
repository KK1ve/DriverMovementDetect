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
#include <utils.h>
using namespace std;
using namespace cv;



class IS
{
    public:
        explicit IS(const string& modelpath, int use_int8 = 0, float nms_thresh_ = 0.5, float conf_thresh_ = 0.6);
        // tuple [frame_id, last_step_result, origin_mat]

        CommonResultSeg pre_process(CommonResultSeg& input);
        CommonResultSeg detect(CommonResultSeg& input);
        CommonResultSeg post_process(CommonResultSeg& input);
        CommonResultSeg vis(CommonResultSeg& input);


    private:
        int batchSize;
        int inpWidth;
        int inpHeight;
        float nmsThresh;
        float confThresh;

        int image_area;

        void generateProposal(const vector<float>& pred, CommonResultSeg &input);

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
