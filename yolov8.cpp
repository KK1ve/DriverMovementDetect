// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#include "yolov8.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

static float sigmod(const float in)
{
    return 1.f / (1.f + expf(-1.f * in));
}

static float softmax(
    const float* src,
    float* dst,
    int length
)
{
    float alpha = -FLT_MAX;
    for (int c = 0; c < length; c++)
    {
        float score = src[c];
        if (score > alpha)
        {
            alpha = score;
        }
    }

    float denominator = 0;
    float dis_sum = 0;
    for (int i = 0; i < length; ++i)
    {
        dst[i] = expf(src[i] - alpha);
        denominator += dst[i];
    }
    for (int i = 0; i < length; ++i)
    {
        dst[i] /= denominator;
        dis_sum += i * dst[i];
    }
    return dis_sum;
}

static void qsort_descent_inplace(std::vector<ObjectYolov8>& faceobjects, int left, int right) {
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j) {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j) {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

    //#pragma omp parallel sections
    {
        //#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<ObjectYolov8>& faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void generate_proposals(
    const ncnn::Mat& feat_blob,
    const float prob_threshold,
    const int num_classes,
    std::vector<std::vector<ObjectYolov8>>& objects
)
{


    const int num_c = feat_blob.c;  //1
    const int num_grid_w = feat_blob.w; //8400
    const int num_grid_h = feat_blob.h; //10



    const int kps_num = (num_grid_h - 4 - num_classes) / 3;
    std::vector<std::vector<float>> table(num_grid_w, std::vector<float>(num_grid_h));


    for (int q = 0; q < feat_blob.c; q++)
    {
        const float* ptr = feat_blob.channel(q);

        for (int y = 0; y < feat_blob.h; y++)
        {
            for (int x = 0; x < feat_blob.w; x++)
            {
                table[x][y] = ptr[x];
            }
            ptr += feat_blob.w;

        }

    }

    for (int y = 0; y < num_grid_w; y++)
    {
        for (int i = 0; i < num_classes; i++) {
            if (table[y][4 + i] >= prob_threshold) {
                std::vector<float> kps;
                for (int k = 0; k < kps_num; k++)
                {
                    float kps_x = table[y][4 + num_classes + k * 3];
                    float kps_y = table[y][4 + num_classes + k * 3 + 1];
                    float kps_s = table[y][4 + num_classes + k * 3 + 2];


                    kps.push_back(kps_x);
                    kps.push_back(kps_y);
                    kps.push_back(kps_s);
                }

                ObjectYolov8 temp_object;
                temp_object.label = i;
                temp_object.prob = table[y][4 + i];
                temp_object.rect.x = table[y][0] - table[y][2] / 2;
                temp_object.rect.y = table[y][1] - table[y][3] / 2;
                temp_object.rect.width = table[y][2];
                temp_object.rect.height = table[y][3];
                printf("box x0:%f, y0:%f, w:%f, h:%f label:%d\n", temp_object.rect.x, temp_object.rect.y, temp_object.rect.width, temp_object.rect.height ,i);

                objects[i].push_back(temp_object);

                break;

            }
        }
        

    }
}

static float clamp(
    float val,
    float min = 0.f,
    float max = 1280.f
)
{
    return val > min ? (val < max ? val : max) : min;
}


typedef struct {
    cv::Rect box;
    float confidence;
    int index;
}BBOX;

static bool cmp_score(BBOX box1, BBOX box2) {
    return box1.confidence > box2.confidence;
}

static float get_iou_value(cv::Rect rect1, cv::Rect rect2)
{
    int xx1, yy1, xx2, yy2;

    xx1 = std::max(rect1.x, rect2.x);
    yy1 = std::max(rect1.y, rect2.y);
    xx2 = std::min(rect1.x + rect1.width - 1, rect2.x + rect2.width - 1);
    yy2 = std::min(rect1.y + rect1.height - 1, rect2.y + rect2.height - 1);

    int insection_width, insection_height;
    insection_width = std::max(0, xx2 - xx1 + 1);
    insection_height = std::max(0, yy2 - yy1 + 1);

    float insection_area, union_area, iou;
    insection_area = float(insection_width) * insection_height;
    union_area = float(rect1.width * rect1.height + rect2.width * rect2.height - insection_area);
    iou = insection_area / union_area;
    return iou;
}


//类似于cv::dnn::NMSBoxes的接口
//input:  boxes: 原始检测框集合;
//input:  confidences：原始检测框对应的置信度值集合
//input:  confThreshold 和 nmsThreshold 分别是 检测框置信度阈值以及做nms时的阈值
//output:  indices  经过上面两个阈值过滤后剩下的检测框的index
static void my_nms_boxes(std::vector<cv::Rect>& boxes, std::vector<float>& confidences, float nmsThreshold,
    std::vector<int>& indices)
{
    BBOX bbox;
    std::vector<BBOX> bboxes;
    int i, j;
    for (i = 0; i < boxes.size(); i++)
    {
        //memset((void*)&bbox, 0x00, sizeof(bbox));
        bbox.box = boxes[i];
        bbox.confidence = confidences[i];
        bbox.index = i;
        bboxes.push_back(bbox);
    }
    sort(bboxes.begin(), bboxes.end(), cmp_score);

    while (true) {
        if (bboxes.size() <= 0) {
            break;
        }
        indices.push_back(bboxes[0].index);
        for (int i = 0; i < bboxes.size(); i++) {
            float iou = get_iou_value(bboxes[0].box, bboxes[i].box);
            if (iou >= nmsThreshold)
            {
                bboxes.erase(bboxes.begin() + i);
                i = i - 1;
            }
        }


    }
    //int updated_size = bboxes.size();
    //for (i = 0; i < updated_size; i++)
    //{
    //    indices.push_back(bboxes[i].index);
    //    for (j = i + 1; j < updated_size; j++)
    //    {
    //        float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
    //        if (iou >= nmsThreshold)
    //        {
    //            bboxes.erase(bboxes.begin() + j);
    //            j = j - 1;
    //            updated_size = bboxes.size();
    //        }
    //    }
    //}
}

static float intersection_area(const ObjectYolov8& a, const ObjectYolov8& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}


static void non_max_suppression(int num_cls, std::vector<std::vector<ObjectYolov8>>& proposals, std::vector<ObjectYolov8>& results,
    int orin_h, int orin_w, float dh = 0, float dw = 0, float ratio_h = 1.0f,
    float ratio_w = 1.0f, float conf_thres = 0.25f, float iou_thres = 0.5f)

{
    results.clear();
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    std::vector<std::vector<float>> kpss;

    for (int la = 0; la < num_cls; la++) {
        if (proposals[la].size() <= 0) {
            continue;
        }
        bboxes.clear();
        scores.clear();
        labels.clear();
        kpss.clear();
        indices.clear();

        for (int i = 0; i < proposals[la].size(); i++) {
            bboxes.push_back(proposals[la][i].rect);
            scores.push_back(proposals[la][i].prob);
            labels.push_back(proposals[la][i].label);
        }
        my_nms_boxes(
            bboxes,
            scores,
            iou_thres,
            indices
        );
        for (int i : indices)
        {

            auto& bbox = proposals[la][i].rect;

            float& score = proposals[la][i].prob;
            int& label = proposals[la][i].label;

            float x0 = clamp((bbox.x - dw) / ratio_h, 0.f, orin_w);
            float y0 = clamp((bbox.y - dh) / ratio_h, 0.f, orin_h);
            float width = clamp(bbox.width, 0.f, orin_w);
            float height = clamp(bbox.height, 0.f, orin_w);


            ObjectYolov8 obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = width;
            obj.rect.height = height;
            obj.prob = score;
            obj.label = label;
            results.push_back(obj);
        }

    }



}

Yolov8::Yolov8()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}


int Yolov8::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, std::vector<std::string> _class_name, bool use_gpu)
{
    yolov8.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolov8.opt = ncnn::Option();

#if NCNN_VULKAN
    yolov8.opt.use_vulkan_compute = use_gpu;
#endif

    yolov8.opt.num_threads = ncnn::get_big_cpu_count();
    yolov8.opt.blob_allocator = &blob_pool_allocator;
    yolov8.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/model_zoo/%s.param", CURRENT_DIR, modeltype);
    sprintf(modelpath, "%s/model_zoo/%s.bin", CURRENT_DIR, modeltype);

    yolov8.load_param(parampath);
    yolov8.load_model(modelpath);

    target_size = _target_size;
    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];
    class_names.clear();
    for (int i = 0; i < _class_name.size(); i++) {
        class_names.push_back(_class_name[i]);
    }
    return 0;
}

int Yolov8::detect(const cv::Mat& rgb, std::vector<ObjectYolov8>& objects, float prob_threshold, float nms_threshold)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // pad to target_size rectangle
    int wpad = std::abs(w - target_size);
    int hpad = std::abs(h - target_size);

    int top = hpad / 2;
    int bottom = hpad - hpad / 2;
    int left = wpad / 2;
    int right = wpad - wpad / 2;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in,
        in_pad,
        top,
        bottom,
        left,
        right,
        ncnn::BORDER_CONSTANT,
        114.f);

    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov8.create_extractor();

    ex.input("in0", in_pad);


    ncnn::Mat out;
    ex.extract("out0", out);

    const int num_class = class_names.size();

    std::vector<std::vector<ObjectYolov8>> proposals(num_class, std::vector<ObjectYolov8>(0));

    generate_proposals(out, prob_threshold, num_class, proposals);

    non_max_suppression(num_class, proposals, objects,
        height, width, hpad / 2, wpad / 2,
        scale, scale, prob_threshold, nms_threshold);

    return 0;
}

int Yolov8::draw(cv::Mat& rgb, const std::vector<ObjectYolov8>& objects)
{
    static const unsigned char colors[19][3] = {
        { 54,  67, 244},
        { 99,  30, 233},
        {176,  39, 156},
        {183,  58, 103},
        {181,  81,  63},
        {243, 150,  33},
        {244, 169,   3},
        {212, 188,   0},
        {136, 150,   0},
        { 80, 175,  76},
        { 74, 195, 139},
        { 57, 220, 205},
        { 59, 235, 255},
        {  7, 193, 255},
        {  0, 152, 255},
        { 34,  87, 255},
        { 72,  85, 121},
        {158, 158, 158},
        {139, 125,  96}
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const ObjectYolov8& obj = objects[i];

        //         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
        //                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, obj.rect, cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
    }

    return 0;
}