//
// Created by 171153 on 2024/8/7.
//
#include <iostream>
#include <vector>
#include<cmath>
#include "test.h"

#include <algorithm>
// 递归函数，将多维vector展平为一维vector
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
    vector<float> a;
    a.resize(2);
    float arr[] = {2.0f, 4.0f, 6.0f, 8.0f};
    float* z = arr;
    // std::transform(z, z + 4, a.begin(), [](float x) {
    //        return x / 2;
    //    });
    std::copy(z, z + 4, a.begin());

    for (auto& c : a)
    {
        cout << c << endl;
    }



}
