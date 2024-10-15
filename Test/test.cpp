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
    int m_num_channels = 100;
    auto m_input_f32 = new float[m_num_channels];
    float *channel_base = m_input_f32;
    channel_base += 100;
    for (int i = 0; i < m_num_channels; i++) {
        channel_base += 512 * 1024;
    }

}
