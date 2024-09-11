//
// Created by 171153 on 2024/8/7.
//
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>
#include "test.h"
// 递归函数，将多维vector展平为一维vector

using namespace cv;
using namespace std;


int main() {
    Mat mat = (Mat_<int>(3, 3) << 1, 2, 3,
                                  4, 5, 6,
                                  7, 8, 9);

    cout << mat.at<int>(1,0) << endl;

    return 0;

}
