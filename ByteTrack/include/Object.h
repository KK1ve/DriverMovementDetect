#pragma once

#include "Rect.h"

namespace byte_track
{
struct Object
{
    Rect<float> rect;
    int label;
    float prob;
    std::vector<float> action;

    Object(const Rect<float> &_rect,
           const int &_label,
           const float &_prob,
           const std::vector<float> &_action);
};
}