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
    vector<float> a = {3,2,1,7};
    auto argmax_x = std::distance(a.begin(),
    std::min_element(a.begin(),
        a.end()));
    cout << argmax_x << endl;
    for (auto& z : argmax_x)
    {
        cout << z << endl;
    }

}
