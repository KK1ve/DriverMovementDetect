//
// Created by 171153 on 2024/8/12.
//

#ifndef INSTANCESEGMENTATION_H
#define INSTANCESEGMENTATION_H


#include <opencv2/highgui.hpp>
#include <cmath>
#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>



using namespace std;
using namespace cv;
using namespace Ort;

template <
    class B,
    class A,
    template <class...> class Container,
    class... extras>
void unflat_transfer(Container<B, extras...>& target, const std::vector<A>& source, size_t& index, const std::vector<size_t>& dimensions, size_t dim = 0)
{
    target.resize(dimensions[dim]);
    for(auto& elem : target)
    {
        if constexpr(std::is_same<B, A>::value)
        {
            elem = source[index++];
        }
        else
        {
            unflat_transfer(elem, source, index, dimensions, dim + 1);
        }
    }
}

// 重载函数用于将一维向量转为多维容器
template <
    class B,
    template <class...> class Container,
    class A>
Container<B> unflat_transfer(const std::vector<A>& source, const std::vector<size_t>& dimensions)
{
    Container<B> target;
    size_t index = 0;
    unflat_transfer(target, source, index, dimensions);
    return target;
}


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

        Session* ort_session = nullptr;


        void generate_proposal(const float *pred, vector<ObjectSeg> &boxes);
        Env env = Env(ORT_LOGGING_LEVEL_VERBOSE, "IS Instance Segmentation");

        SessionOptions sessionOptions = SessionOptions();
        const OrtApi& api = Ort::GetApi();
        vector<char *> input_names;
        vector<char *> output_names;
        vector<vector<int64_t>> input_node_dims;  // >=1 outputs
        vector<vector<int64_t>> output_node_dims; // >=1 outputs
        Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

        const char* labels[3] = {"rail-raised", "rail-track", "unidentified"};
        const int num_class = 3;
        const vector<vector<int>> mat_colos = {
            {0, 0, 255},
            {255, 255, 102},
            {0, 0, 0}
        };

};




#endif //INSTANCESEGMENTATION_H
