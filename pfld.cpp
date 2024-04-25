
#include "pfld.h"
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

static void generate_proposals(
    int stride,
    const ncnn::Mat& feat_blob,
    const float prob_threshold,
    std::vector<Object_pose_pfld>& objects
)
{
    const int reg_max = 16;
    float dst[16];
    const int num_w = feat_blob.w;
    const int num_grid_y = feat_blob.c;
    const int num_grid_x = feat_blob.h;
    const int kps_num = 17;

    const int num_class = num_w - 4 * reg_max;

    const int clsid = 0;

    for (int i = 0; i < num_grid_y; i++)
    {
        for (int j = 0; j < num_grid_x; j++)
        {
            const float* matat = feat_blob.channel(i).row(j);

            float score = matat[0];
            score = sigmod(score);
            if (score < prob_threshold)
            {
                continue;
            }

            float x0 = j + 0.5f - softmax(matat + 1, dst, 16);
            float y0 = i + 0.5f - softmax(matat + (1 + 16), dst, 16);
            float x1 = j + 0.5f + softmax(matat + (1 + 2 * 16), dst, 16);
            float y1 = i + 0.5f + softmax(matat + (1 + 3 * 16), dst, 16);

            x0 *= stride;
            y0 *= stride;
            x1 *= stride;
            y1 *= stride;

            std::vector<float> kps;
            for (int k = 0; k < kps_num; k++)
            {
                float kps_x = (matat[1 + 64 + k * 3] * 2.f + j) * stride;
                float kps_y = (matat[1 + 64 + k * 3 + 1] * 2.f + i) * stride;
                float kps_s = sigmod(matat[1 + 64 + k * 3 + 2]);

                kps.push_back(kps_x);
                kps.push_back(kps_y);
                kps.push_back(kps_s);
            }

            Object_pose_pfld obj;
            obj.rect.x = x0;
            obj.rect.y = y0;
            obj.rect.width = x1 - x0;
            obj.rect.height = y1 - y0;
            obj.label = 0;
            obj.prob = score;
            obj.kps = kps;
            objects.push_back(obj);
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
static void my_nms_boxes(std::vector<cv::Rect>& boxes, std::vector<float>& confidences, float confThreshold, float nmsThreshold, std::vector<int>& indices)
{
    BBOX bbox;
    std::vector<BBOX> bboxes;
    int i, j;
    for (i = 0; i < boxes.size(); i++)
    {
        bbox.box = boxes[i];
        bbox.confidence = confidences[i];
        bbox.index = i;
        bboxes.push_back(bbox);
    }
    sort(bboxes.begin(), bboxes.end(), cmp_score);

    int updated_size = bboxes.size();
    for (i = 0; i < updated_size; i++)
    {
        if (bboxes[i].confidence < confThreshold)
            continue;
        indices.push_back(bboxes[i].index);
        for (j = i + 1; j < updated_size; j++)
        {
            float iou = get_iou_value(bboxes[i].box, bboxes[j].box);
            if (iou > nmsThreshold)
            {
                bboxes.erase(bboxes.begin() + j);
                j = j - 1;
                updated_size = bboxes.size();
            }
        }
    }
}


static void non_max_suppression(std::vector<Object_pose_pfld>& proposals, std::vector<Object_pose_pfld>& results,
    int orin_h, int orin_w, float dh = 0, float dw = 0, float ratio_h = 1.0f,
    float ratio_w = 1.0f, float conf_thres = 0.25f, float iou_thres = 0.65f)

{
    results.clear();
    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;
    std::vector<std::vector<float>> kpss;

    for (auto& pro : proposals)
    {
        bboxes.push_back(pro.rect);
        scores.push_back(pro.prob);
        labels.push_back(pro.label);
        kpss.push_back(pro.kps);
    }

    //cv::dnn::NMSBoxes(
    //        bboxes,
    //        scores,
    //        conf_thres,
    //        iou_thres,
    //        indices
    //);
    my_nms_boxes(
        bboxes,
        scores,
        conf_thres,
        iou_thres,
        indices
    );

    for (auto i : indices)
    {
        auto& bbox = bboxes[i];
        float x0 = bbox.x;
        float y0 = bbox.y;
        float x1 = bbox.x + bbox.width;
        float y1 = bbox.y + bbox.height;
        float& score = scores[i];
        int& label = labels[i];

        x0 = (x0 - dw) / ratio_w;
        y0 = (y0 - dh) / ratio_h;
        x1 = (x1 - dw) / ratio_w;
        y1 = (y1 - dh) / ratio_h;

        x0 = clamp(x0, 0.f, orin_w);
        y0 = clamp(y0, 0.f, orin_h);
        x1 = clamp(x1, 0.f, orin_w);
        y1 = clamp(y1, 0.f, orin_h);

        Object_pose_pfld obj;
        obj.rect.x = x0;
        obj.rect.y = y0;
        obj.rect.width = x1 - x0;
        obj.rect.height = y1 - y0;
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

pfld_pose::pfld_pose()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int pfld_pose::load(const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu) {

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
    return 0;
}

//int Yolo_pose::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
//{
//    yolo.clear();
//    blob_pool_allocator.clear();
//    workspace_pool_allocator.clear();
//
//    ncnn::set_cpu_powersave(2);
//    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());
//
//    yolo.opt = ncnn::Option();
//
//#if NCNN_VULKAN
//    yolo.opt.use_vulkan_compute = use_gpu;
//#endif
//
//    yolo.opt.num_threads = ncnn::get_big_cpu_count();
//    yolo.opt.blob_allocator = &blob_pool_allocator;
//    yolo.opt.workspace_allocator = &workspace_pool_allocator;
//
//    char parampath[256];
//    char modelpath[256];
//    sprintf(parampath, "yolov8%s.param", modeltype);
//    sprintf(modelpath, "yolov8%s.bin", modeltype);
//
//    yolo.load_param(mgr, parampath);
//    yolo.load_model(mgr, modelpath);
//
//    target_size = _target_size;
//    mean_vals[0] = _mean_vals[0];
//    mean_vals[1] = _mean_vals[1];
//    mean_vals[2] = _mean_vals[2];
//    norm_vals[0] = _norm_vals[0];
//    norm_vals[1] = _norm_vals[1];
//    norm_vals[2] = _norm_vals[2];
//
//    return 0;
//}
static void visualize(const char* title, const ncnn::Mat& m)
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
    cv::waitKey(0);
}


static void pretty_print(const ncnn::Mat& m)
{
    printf("%d ", m.c);
    printf("\n");
    printf("%d ", m.h);
    printf("\n");
    printf("%d ", m.w);
    printf("\n");
    for (int q = 0; q < m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int y = 0; y < m.h; y++)
        {
            for (int x = 0; x < m.w; x++)
            {
                printf("%f ", ptr[x]);
            }
            ptr += m.w;
            printf("\n");
        }
        printf("------------------------\n");
    }
}

int pfld_pose::detect(const cv::Mat& bgr, ncnn::Mat& result_mat, float prob_threshold, float nms_threshold) {
    int img_w = bgr.cols;
    int img_h = bgr.rows;

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(bgr.data, ncnn::Mat::PIXEL_BGR2RGB, img_w, img_h, 112, 112);

    ncnn::Extractor ex = yolo.create_extractor();

    ex.input("input_1", in);
    ex.extract("415", result_mat);
    //std::vector<Object_pose_pfld> objects8;
    //generate_proposals(8, out, prob_threshold, objects8);

    //proposals.insert(proposals.end(), objects8.begin(), objects8.end());

    return 0;
}


int pfld_pose::draw(cv::Mat& rgb, ncnn::Mat& out) {

    const int num_landmarks = 106 * 2;
    float landmarks[num_landmarks];

    int w = rgb.cols;
    int h = rgb.rows;

    for (int j = 0; j < out.w; j++)
    {
        landmarks[j] = out[j];
    }
    for (int i = 0; i < num_landmarks / 2; i++) {
        cv::circle(rgb, cv::Point(landmarks[i * 2] * w , landmarks[i * 2 + 1] * h ),
            2, cv::Scalar(0, 0, 255), -1);
    }


    return 0;
}



