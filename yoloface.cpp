//
// Created by wangke on 2023/5/15.
//

#include "yoloface.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cpu.h>

#define MAX_STRIDE 32 // if yolov8-p6 model modify to 64

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

static void qsort_descent_inplace(std::vector<Object_Face>& faceobjects, int left, int right) {
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

static void qsort_descent_inplace(std::vector<Object_Face>& faceobjects) {
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void generate_proposals(
    const ncnn::Mat& feat_blob,
    const float prob_threshold,
    const int num_classes,
    std::vector<std::vector<Object_Face>>& objects
)
{


    const int num_c = feat_blob.c;  //1
    const int num_grid_w = feat_blob.w; //8400
    const int num_grid_h = feat_blob.h; //209


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
        for (int i = 0; i < num_classes; i++){
            if (table[y][4 + i] >= prob_threshold){
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

                Object_Face temp_object;
                temp_object.kps = kps;
                temp_object.label = i;
                temp_object.prob = table[y][4 + i];
                temp_object.rect.x = table[y][0] - table[y][2] / 2;
                temp_object.rect.y = table[y][1] - table[y][3] / 2;
                temp_object.rect.width = table[y][2];
                temp_object.rect.height = table[y][3];
                //printf("box x0:%f, y0:%f, w:%f, h:%f label:%d\n", temp_object.rect.x, temp_object.rect.y, temp_object.rect.width, temp_object.rect.height ,i);

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

static float intersection_area(const Object_Face& a, const Object_Face& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}


static void non_max_suppression(int num_cls, std::vector<std::vector<Object_Face>>& proposals, std::vector<Object_Face>& results,
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
            kpss.push_back(proposals[la][i].kps);
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


            Object_Face obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = width;
            obj.rect.height = height;
            obj.prob = score;
            obj.label = label;
            obj.kps = kpss[i];
            for (int n = 0; n < obj.kps.size(); n += 3)
            {
                obj.kps[n] = clamp((obj.kps[n] - dw) / ratio_w, 0.f, orin_w);
                obj.kps[n + 1] = clamp((obj.kps[n + 1] - dh) / ratio_h, 0.f, orin_h);
            }
            results.push_back(obj);
        }

    }



}

Yolo_Face::Yolo_Face()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Yolo_Face::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, std::vector<std::string> _class_name, bool use_gpu) {

    yolo.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    yolo.opt = ncnn::Option();

#if NCNN_VULKAN
    yolo.opt.use_vulkan_compute = use_gpu;
#endif

    yolo.opt.num_threads = ncnn::get_big_cpu_count();
    yolo.opt.blob_allocator = &blob_pool_allocator;
    yolo.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/model_zoo/%s.param", CURRENT_DIR, modeltype);
    sprintf(modelpath, "%s/model_zoo/%s.bin", CURRENT_DIR, modeltype);

    yolo.load_param(parampath);
    yolo.load_model(modelpath);

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
    num_classes = class_names.size();

    return 0;
}


void visualize(const char* title, const ncnn::Mat& m)
{
    std::vector<cv::Mat> normed_feats(m.c);

    for (int i = 0; i < m.c; i++)
    {
        cv::Mat tmp(m.h, m.w, CV_32FC1, (void*)(const float*)m.channel(i));

        cv::normalize(tmp, normed_feats[i], 0, 255, cv::NORM_MINMAX, CV_8U);

        cv::cvtColor(normed_feats[i], normed_feats[i], cv::COLOR_GRAY2BGR);

        // check NaN
        for (int y = 0; y < m.h; y++)
        {
            const float* tp = tmp.ptr<float>(y);
            uchar* sp = normed_feats[i].ptr<uchar>(y);
            for (int x = 0; x < m.w; x++)
            {
                float v = tp[x];
                if (v != v)
                {
                    sp[0] = 0;
                    sp[1] = 0;
                    sp[2] = 255;
                }

                sp += 3;
            }
        }
    }

    int tw = m.w < 10 ? 32 : m.w < 20 ? 16 : m.w < 40 ? 8 : m.w < 80 ? 4 : m.w < 160 ? 2 : 1;
    int th = (m.c - 1) / tw + 1;

    cv::Mat show_map(m.h * th, m.w * tw, CV_8UC3);
    show_map = cv::Scalar(127);

    // tile
    for (int i = 0; i < m.c; i++)
    {
        int ty = i / tw;
        int tx = i % tw;

        normed_feats[i].copyTo(show_map(cv::Rect(tx * m.w, ty * m.h, m.w, m.h)));
    }

    cv::resize(show_map, show_map, cv::Size(0, 0), 2, 2, cv::INTER_NEAREST);
    cv::imshow(title, show_map);
}


float euclidean_distance(float x1, float x2, float y1, float y2) {
    return sqrt((pow(abs(x1 - x2), 2)) + (pow(abs(y1 - y2), 2)));
}


int Yolo_Face::detect(const cv::Mat& bgr, std::vector<Object_Face>& objects, float prob_threshold, float nms_threshold) {
    int img_w = bgr.cols;
    int img_h = bgr.rows;
    // letterbox pad to multiple of MAX_STRIDE
    int w = img_w;
    int h = img_h;
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

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, w, h);


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

    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("in0", in_pad);



    ncnn::Mat out;
    ex.extract("out0", out);
        
    std::vector<std::vector<Object_Face>> proposals(num_classes, std::vector<Object_Face>(0));
    generate_proposals(out, prob_threshold, num_classes, proposals);

    //qsort_descent_inplace(proposals);

    non_max_suppression(num_classes, proposals, objects,
        img_h, img_w, hpad / 2, wpad / 2,
        scale, scale, prob_threshold, nms_threshold);
    //for (Object_Face& obj : objects) {
    //    printf("%s - 置信度: %f\n",class_names[obj.label], obj.prob);
    //}

    return 0;
}




int Yolo_Face::draw(cv::Mat& rgb, std::vector<Object_Face>& objects) {

    const std::vector<std::vector<unsigned int>> KPS_COLORS =
    { {0,   255, 0},
      {0,   255, 0},
      {0,   255, 0},
      {0,   255, 0},
      {0,   255, 0},
      {255, 128, 0},
      {255, 128, 0},
      {255, 128, 0},
      {255, 128, 0},
      {255, 128, 0},
      {255, 128, 0},
      {51,  153, 255},
      {51,  153, 255},
      {51,  153, 255},
      {51,  153, 255},
      {51,  153, 255},
      {51,  153, 255} };

    const std::vector<std::vector<unsigned int>> SKELETON = { {16, 14},
                                                              {14, 12},
                                                              {17, 15},
                                                              {15, 13},
                                                              {12, 13},
                                                              {6,  12},
                                                              {7,  13},
                                                              {6,  7},
                                                              {6,  8},
                                                              {7,  9},
                                                              {8,  10},
                                                              {9,  11},
                                                              {2,  3},
                                                              {1,  2},
                                                              {1,  3},
                                                              {2,  4},
                                                              {3,  5},
                                                              {4,  6},
                                                              {5,  7} };

    const std::vector<std::vector<unsigned int>> LIMB_COLORS = { {51,  153, 255},
                                                                 {51,  153, 255},
                                                                 {51,  153, 255},
                                                                 {51,  153, 255},
                                                                 {255, 51,  255},
                                                                 {255, 51,  255},
                                                                 {255, 51,  255},
                                                                 {255, 128, 0},
                                                                 {255, 128, 0},
                                                                 {255, 128, 0},
                                                                 {255, 128, 0},
                                                                 {255, 128, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0},
                                                                 {0,   255, 0} };



    cv::Mat res = rgb;
    for (Object_Face& obj : objects) {
        cv::rectangle(res, obj.rect, { 0, 0, 255 }, 2);
        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label].c_str(), obj.prob * 100);
        if (obj.label == 0) {
            sprintf(text, "%s pose: %s", text, obj.pose_name);
        }
        else {
            char face_text[256];
            sprintf(face_text, "infos: ");
            std::vector<int> right_eyes = { 0, 2, 4, 6, 8, 10};
            std::vector<int> left_eyes = { 1, 3, 5, 7, 9, 11};
            std::vector<int> mouth = { 12, 14, 15, 13, 17, 16};

            float EAR = (euclidean_distance(obj.kps[right_eyes[1] * 3 + 0], 
                obj.kps[right_eyes[5] * 3 + 0], obj.kps[right_eyes[1] * 3 + 1], 
                obj.kps[right_eyes[5] * 3 + 1]) + euclidean_distance(obj.kps[right_eyes[2] * 3 + 0],
                    obj.kps[right_eyes[4] * 3 + 0], obj.kps[right_eyes[2] * 3 + 1],
                    obj.kps[right_eyes[4] * 3 + 1])) / (2 * euclidean_distance(obj.kps[right_eyes[0] * 3 + 0],
                        obj.kps[right_eyes[3] * 3 + 0], obj.kps[right_eyes[0] * 3 + 1],
                        obj.kps[right_eyes[3] * 3 + 1]));

            sprintf(face_text, "%s RIGHT: %.2f", face_text, EAR);


            EAR = (euclidean_distance(obj.kps[left_eyes[1] * 3 + 0],
                obj.kps[left_eyes[5] * 3 + 0], obj.kps[left_eyes[1] * 3 + 1],
                obj.kps[left_eyes[5] * 3 + 1]) + euclidean_distance(obj.kps[left_eyes[2] * 3 + 0],
                    obj.kps[left_eyes[4] * 3 + 0], obj.kps[left_eyes[2] * 3 + 1],
                    obj.kps[left_eyes[4] * 3 + 1])) / (2 * euclidean_distance(obj.kps[left_eyes[0] * 3 + 0],
                        obj.kps[left_eyes[3] * 3 + 0], obj.kps[left_eyes[0] * 3 + 1],
                        obj.kps[left_eyes[3] * 3 + 1]));

            sprintf(face_text, "%s LEFT: %.2f", face_text, EAR);


            EAR = (euclidean_distance(obj.kps[mouth[1] * 3 + 0],
                obj.kps[mouth[5] * 3 + 0], obj.kps[mouth[1] * 3 + 1],
                obj.kps[mouth[5] * 3 + 1]) + euclidean_distance(obj.kps[mouth[2] * 3 + 0],
                    obj.kps[mouth[4] * 3 + 0], obj.kps[mouth[2] * 3 + 1],
                    obj.kps[mouth[4] * 3 + 1])) / (2 * euclidean_distance(obj.kps[mouth[0] * 3 + 0],
                        obj.kps[mouth[3] * 3 + 0], obj.kps[mouth[0] * 3 + 1],
                        obj.kps[mouth[3] * 3 + 1]));

            sprintf(face_text, "%s MOUTH: %.2f", face_text, EAR);

            int baseLine = 0;
            cv::Size label_size = cv::getTextSize(face_text, cv::FONT_HERSHEY_SIMPLEX,
                0.4, 1, &baseLine);

            int x = (int)obj.rect.x;
            int y = (int)obj.rect.y + 1;

            if (y > res.rows)
                y = res.rows;

            cv::putText(res, face_text, cv::Point(x, y - label_size.height),
                cv::FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 }, 1);

        }
        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX,
            0.4, 1, &baseLine);

        int x = (int)obj.rect.x;
        int y = (int)obj.rect.y + 1;

        if (y > res.rows)
            y = res.rows;

        cv::rectangle(res, cv::Rect(x, y, label_size.width, label_size.height + baseLine),
            { 0, 0, 255 }, -1);

        cv::putText(res, text, cv::Point(x, y + label_size.height),
            cv::FONT_HERSHEY_SIMPLEX, 0.4, { 255, 255, 255 }, 1);

        std::vector<float>& kps = obj.kps;
        for (int k = 0; k < num_points; k++)
        {
            int kps_x= std::round(kps[k * 3]);
            int kps_y=std::round(kps[k * 3 + 1]);
            float kps_s = kps[k * 3 + 2];


            if (kps_s > 0.5f)
            {
                cv::Scalar kps_color = cv::Scalar(KPS_COLORS[(int)k % 17][0], KPS_COLORS[(int)k % 17][1], KPS_COLORS[(int)k % 17][2]);
                cv::circle(res, { kps_x, kps_y }, 5, kps_color, -1);
            }

        }
        //for (int k = 0; k < SKELETON.size(); k++) {
        //    auto& ske = SKELETON[k];
        //    int pos1_x = std::round(kps[(ske[0] - 1) * 3]);
        //    int pos1_y = std::round(kps[(ske[0] - 1) * 3 + 1]);

        //    int pos2_x = std::round(kps[(ske[1] - 1) * 3]);
        //    int pos2_y = std::round(kps[(ske[1] - 1) * 3 + 1]);

        //    float pos1_s = kps[(ske[0] - 1) * 3 + 2];
        //    float pos2_s = kps[(ske[1] - 1) * 3 + 2];

        //    if (pos1_s > 0.5f && pos2_s > 0.5f)
        //    {
        //        cv::Scalar limb_color = cv::Scalar(LIMB_COLORS[k][0], LIMB_COLORS[k][1], LIMB_COLORS[k][2]);
        //        cv::line(res, { pos1_x, pos1_y }, { pos2_x, pos2_y }, limb_color, 2);
        //    }
        //}

    }

    return 0;
}

