//
// Created by 171153 on 2024/10/18.
//

#ifndef UTILS_H
#define UTILS_H


#include <numeric>

#include "opencv2/opencv.hpp"
#include "vector"
typedef struct
{
    int xmin;
    int ymin;
    int xmax;
    int ymax;
} Bbox;

struct ObjectPose
{
    // x左上角 y左上角
    cv::Rect_<float> rect;
    int label;
    float prob;
    std::vector<float> kps;
    std::vector<float> action_prob;
    unsigned long track_id = -1;
};

static float euclidean_distance(float x1, float x2, float y1, float y2) {
    return sqrt((pow(abs(x1 - x2), 2)) + (pow(abs(y1 - y2), 2)));
}


static cv::Mat letterbox(const cv::Mat &src, int h, int w)
{

    const int in_w = src.cols; // width
    const int in_h = src.rows; // height
    const int tar_w = w;
    const int tar_h = h;
    const float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    const int inside_w = round(in_w * r);
    const int inside_h = round(in_h * r);
    cv::Mat resize_img;

    cv::resize(src, resize_img, cv::Size(inside_w, inside_h));

    const int top = (tar_h - inside_h) / 2;
    const int bottom = tar_h - inside_h - top;
    const int left = (tar_w - inside_w) / 2;
    const int right = tar_w - inside_w - left;
    cv::copyMakeBorder(resize_img, resize_img, top, bottom, left, right, 0, cv::Scalar(114, 114, 114));

    return resize_img;
}

static cv::Mat un_letterbox(const cv::Mat &src, int h, int w)
{
    const int tar_w = src.cols;
    const int tar_h = src.rows;
    const float r = std::min(float(tar_h) / h, float(tar_w) / w);
    const int inside_w = round(float(w) * r);
    const int inside_h = round(float(h) * r);
    int padd_w = tar_w - inside_w;
    int padd_h = tar_h - inside_h;

    padd_w = padd_w / 2;
    padd_h = padd_h / 2;

    const int top = int(round(padd_h - 0.1));
    const int bottom = tar_h - inside_h - top;
    const int left = int(round(padd_w - 0.1));
    const int right = tar_w - inside_w - left;

    cv::Mat temp_mat = src(cv::Rect(left, top, src.cols - left - right, src.rows - top - bottom)).clone();
    cv::Mat restored_image;
    cv::resize(temp_mat, restored_image, cv::Size(w, h));
    return restored_image;

}

static void un_letterbox_obj(const std::vector<ObjectPose>& points, int origin_w, int origin_h, int after_w, int after_h)
{
    int in_w = origin_w; // width
    int in_h = origin_h; // height
    int tar_w = after_w;
    int tar_h = after_h;
    float r = std::min(float(tar_h) / in_h, float(tar_w) / in_w);
    int inside_w = round(in_w * r);
    int inside_h = round(in_h * r);
    int padd_w = tar_w - inside_w;
    int padd_h = tar_h - inside_h;




}
static void softmax(float *x, int row, int column)
{
    for (int j = 0; j < row; ++j)
    {
        float max = 0.0;
        float sum = 0.0;
        for (int k = 0; k < column; ++k)
            if (max < x[k + j*column])
                max = x[k + j*column];
        for (int k = 0; k < column; ++k)
        {
            x[k + j*column] = exp(x[k + j*column] - max);    // prevent data overflow
            sum += x[k + j*column];
        }
        for (int k = 0; k < column; ++k) x[k + j*column] /= sum;
    }
}   //row*column

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}


static float GetIoU(const cv::Rect box1, const cv::Rect box2)
{
    int x1 = std::max(box1.x, box2.x);
    int y1 = std::max(box1.y, box2.y);
    int x2 = std::min(box1.x + box1.width, box2.x + box2.width);
    int y2 = std::min(box1.y + box1.height, box2.y + box2.height);
    int w = std::max(0, x2 - x1);
    int h = std::max(0, y2 - y1);
    float over_area = w * h;
    if (over_area == 0)
        return 0.0;
    float union_area = (box1.x + box1.width - box1.x) * (box1.y + box1.height - box1.y) + (box1.x + box1.width - box2.x) * (box2.y + box2.height - box2.y) - over_area;
    return over_area / union_area;
}

static std::vector<int> TopKIndex(const std::vector<float> &vec, int topk)
{
    std::vector<int> topKIndex;
    topKIndex.clear();

    std::vector<size_t> vec_index(vec.size());
    std::iota(vec_index.begin(), vec_index.end(), 0);

    std::sort(vec_index.begin(), vec_index.end(), [&vec](size_t index_1, size_t index_2)
    { return vec[index_1] > vec[index_2]; });

    int k_num = std::min<int>(vec.size(), topk);

    for (int i = 0; i < k_num; ++i)
    {
        topKIndex.emplace_back(vec_index[i]);
    }

    return topKIndex;
}

static int sub2ind(const int row, const int col, const int cols, const int rows)
{
    return row * cols + col;
}

static void ind2sub(const int sub, const int cols, const int rows, int &row, int &col)
{
    row = sub / cols;
    col = sub % cols;
}

static bool isZero(int num) { return num == 0; }

static bool cmp_score(const ObjectPose& box1, const ObjectPose& box2) {
    // if (box1.prob == box2.prob)
    // {
    //     if(box1.rect.x == box2.rect.x)
    //     {
    //         return box1.rect.y > box2.rect.y;
    //     }
    //     return box1.rect.x > box2.rect.x;
    // }
    return box1.prob > box2.prob;
}

static std::vector<int> multiclass_nms_class_agnostic(std::vector<ObjectPose>& boxes, const float nms_thresh)
{
    std::sort(boxes.begin(), boxes.end(), cmp_score);
    const int num_box = boxes.size();
    std::vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        std::vector<float> ious(boxes.size());

        #pragma omp parallel for
        for (int j = i + 1; j < boxes.size(); ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }
            ious[j] = GetIoU(boxes[i].rect, boxes[j].rect);
        }
        for (int j = i + 1; j < num_box; ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            if (ious[j] > nms_thresh)
            {
                isSuppressed[j] = true;
            }
        }
    }

    std::vector<int> keep_inds;
    for (int i = 0; i < isSuppressed.size(); i++)
    {
        if (!isSuppressed[i])
        {
            keep_inds.emplace_back(i);
        }
    }
    return keep_inds;
}



#endif //UTILS_H
