//
// Created by 171153 on 2024/8/7.
//
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "test.h"
#include "deepsort.h"
// 递归函数，将多维vector展平为一维vector

using namespace cv;
using namespace std;

void asdc(vector<int>& b)
{
    b.emplace_back(1);
}

void sad(vector<int> &a)
{
    asdc(a);
}

int main() {
    deep_sort::DeepSort DS(R"(/home/lsy/CProject/DriverActionDetect/model_zoo/osnet.onnx)", 128, FEATURE_VECTOR_DIM, 0, 30 * 5);


    return 0;

}
