#include "include/TemporalActionDetection.h"
#include "include/PoseEstimation.h"
#include "include/YOLOV3.h"
#include "include/DWPOSE.h"
#include <iostream>
#include <map>
#include <numeric>
#include "deepsort.h"
#include "BYTETracker.h"

using namespace std;

float get_iou_value(ObjectPose& object_pose, deep_sort::DetectBox& detect_box)
{
    int xx1, yy1, xx2, yy2;

    xx1 = std::max(object_pose.rect.x, detect_box.x1);
    yy1 = std::max(object_pose.rect.y, detect_box.y1);
    xx2 = std::min(object_pose.rect.x + object_pose.rect.width - 1, detect_box.x1 + (detect_box.x2 - detect_box.x1) - 1);
    yy2 = std::min(object_pose.rect.y + object_pose.rect.height - 1, detect_box.y1 + (detect_box.y2 - detect_box.y1) - 1);

    int insection_width, insection_height;
    insection_width = std::max(0, xx2 - xx1 + 1);
    insection_height = std::max(0, yy2 - yy1 + 1);

    float insection_area, union_area, iou;
    insection_area = float(insection_width) * insection_height;
    union_area = float(object_pose.rect.width * object_pose.rect.height + (detect_box.x2 - detect_box.x1) * (detect_box.y2 - detect_box.y1) - insection_area);
    iou = insection_area / union_area;
    return iou;
}

void get_relate_track(vector<ObjectPose>& object_poses, vector<deep_sort::DetectBox>& detect_boxes){

    if(object_poses.size() >= detect_boxes.size()){
        vector<int> related_track_vector_index(detect_boxes.size(), 0);
        for(auto & object_pose : object_poses)
        {
            if (object_pose.label != 0) continue;
            float max_iou = LONG_MIN;
            int relate = -1;
            for(int j = 0; j < detect_boxes.size(); j ++)
            {
                if(related_track_vector_index[j] == 1) continue;
                auto iou = get_iou_value(object_pose, detect_boxes[j]);
                if (iou > max_iou)
                {
                    max_iou = iou;
                    relate = j;
                }

            }
            if(relate == -1) continue;
            related_track_vector_index[relate] = 1;
            object_pose.track_id = detect_boxes[relate].trackID;
        }
    }else
    {
        vector<int> related_object_vector_index(object_poses.size(), 0);
        for(auto & detect_box : detect_boxes)
        {
            float max_iou = LONG_MIN;
            int relate = -1;
            for(int j = 0; j < object_poses.size(); j ++)
            {
                if(related_object_vector_index[j] == 1) continue;
                auto iou = get_iou_value(object_poses[j], detect_box);
                if (iou > max_iou)
                {
                    max_iou = iou;
                    relate = j;
                }

            }
            if(relate == -1) continue;
            related_object_vector_index[relate] = 1;
            object_poses[relate].track_id = detect_box.trackID;
        }


    }


}

int detect(Mat& frame, PE& PENet, TAD& TADNet, deep_sort::DeepSort& DS, VideoWriter& vwriter, int width, int height)
{
    vector<deep_sort::DetectBox> track_objects;
    vector<ObjectPose> object_poses;
    try{
        // vector<Bbox> boxes;
        // vector<float> det_conf;
        // vector<vector<float>> cls_conf;
        // vector<int> keep_inds = TADNet.detect_one_hot(frame, boxes, det_conf, cls_conf); ////keep_inds记录vector里面的有效检测框的序号
        // Mat dstimg = TADNet.vis_one_hot(frame, boxes, det_conf, cls_conf, keep_inds, vis_thresh);

        auto s_time = std::chrono::system_clock::now();
        PENet.detect_multi_hot(frame, object_poses);
        auto e_time = std::chrono::system_clock::now();
        std::chrono::duration<float> diff = e_time - s_time;
        cout << "PE TIME: " << diff.count() << endl;
        if (!object_poses.empty())
        {
            for(auto& o: object_poses)
            {

                float _x = o.rect.x > width ? width : o.rect.x < 0 ? 0 : o.rect.x;
                float _y = o.rect.y > height ? height : o.rect.y < 0 ? 0 : o.rect.y;
                float _width = o.rect.width + _x < width ? o.rect.width : width - _x;
                float _height = o.rect.height + _y < height ? o.rect.height : height - _y;
                o.rect.x = _x;
                o.rect.y = _y;
                o.rect.width = _width;
                o.rect.height = _height;
                if(o.label == 0)
                {
                    deep_sort::DetectBox dd(_x, _y, _x + _width, _y + _height, o.prob, o.label);
                    track_objects.emplace_back(dd);
                }

            }
        }else
        {
            vwriter.write(PENet.vis_multi_hot(frame, object_poses));
            return 1;
        }
        s_time = std::chrono::system_clock::now();
        DS.sort(frame, track_objects);
        e_time = std::chrono::system_clock::now();
        diff = e_time - s_time;
        cout << "TRACK TIME: " << diff.count() << endl;
        if (!track_objects.empty())
        {
            std::map<unsigned long, Mat> track_imgs;

            get_relate_track(object_poses, track_objects);

            for(auto& o: object_poses)
            {
                track_imgs[o.track_id] = frame(o.rect).clone();
            }

            auto s_time = std::chrono::system_clock::now();
            std::map<unsigned long, vector<float>> action_result;
            // auto action_result = TADNet.detect_multi_hot(track_imgs);
            auto e_time = std::chrono::system_clock::now();
            std::chrono::duration<float> diff = e_time - s_time;
            cout << "TAD TIME: " << diff.count() << endl;
            for(auto& o: object_poses)
            {
                if(o.label == 0)
                {
                    o.action_prob = action_result[o.track_id];
                }

            }

            Mat pedstimg = PENet.vis_multi_hot(frame, object_poses, true);

            vwriter.write(pedstimg);
            /*imshow("Detect", dstimg);
            if (cv::waitKey(1) > 0) {
                break;
            };*/

        }else
        {
            vwriter.write(PENet.vis_multi_hot(frame, object_poses));
        }

    }catch (exception& e)
    {
        cout << e.what() << endl;
        return 0;
    }
    return 1;
}

int detect(Mat& frame, PE& PENet, TAD& TADNet, byte_track::BYTETracker& BT, VideoWriter& vwriter, int width, int height)
{
    vector<deep_sort::DetectBox> track_objects;
    vector<ObjectPose> object_poses;
    try{
        // vector<Bbox> boxes;
        // vector<float> det_conf;
        // vector<vector<float>> cls_conf;
        // vector<int> keep_inds = TADNet.detect_one_hot(frame, boxes, det_conf, cls_conf); ////keep_inds记录vector里面的有效检测框的序号
        // Mat dstimg = TADNet.vis_one_hot(frame, boxes, det_conf, cls_conf, keep_inds, vis_thresh);

        auto s_time = std::chrono::system_clock::now();
        PENet.detect_multi_hot(frame, object_poses);
        auto e_time = std::chrono::system_clock::now();
        std::chrono::duration<float> diff = e_time - s_time;
        cout << "PE TIME: " << diff.count() << endl;
        vector<byte_track::Object> track_objects;

        if (!object_poses.empty())
        {
            for(auto& o: object_poses)
            {
                if (o.label != 0) continue;
                byte_track::Rect<float> r(o.rect.x, o.rect.y, o.rect.width, o.rect.height);
                byte_track::Object ob(r, o.label, o.prob);
                track_objects.emplace_back(ob);

            }
        }else
        {
            vwriter.write(PENet.vis_multi_hot(frame, object_poses));
            return 1;
        }
        s_time = std::chrono::system_clock::now();
        auto track_data = BT.update(track_objects);
        e_time = std::chrono::system_clock::now();
        diff = e_time - s_time;
        cout << "TRACK TIME: " << diff.count() << endl;
        if (!track_objects.empty())
        {
            std::map<unsigned long, Mat> track_imgs;
            for (int i = 0; i < track_data.size(); i ++)
            {
                int _x = track_data[i]->getRect().x() > width ? width : track_data[i]->getRect().x() < 0 ? 0 : track_data[i]->getRect().x();
                int _y = track_data[i]->getRect().y() > height ? height : track_data[i]->getRect().y() < 0 ? 0 : track_data[i]->getRect().y();
                int _width = track_data[i]->getRect().width() > width - _x ? width - _x : track_data[i]->getRect().width();
                int _height = track_data[i]->getRect().height() > height - _y ? height - _y : track_data[i]->getRect().height();

                object_poses[i].track_id = track_data[i]->getTrackId();
                track_imgs[track_data[i]->getTrackId()] = frame(cv::Rect(
                    _x,
                    _y,
                    _width,
                    _height));
            }

            auto s_time = std::chrono::system_clock::now();
            std::map<unsigned long, vector<float>> action_result;
            // auto action_result = TADNet.detect_multi_hot(track_imgs);
            auto e_time = std::chrono::system_clock::now();
            std::chrono::duration<float> diff = e_time - s_time;
            cout << "TAD TIME: " << diff.count() << endl;
            for(auto& o: object_poses)
            {
                if(o.label == 0)
                {
                    o.action_prob = action_result[o.track_id];
                }

            }

            // Mat pedstimg = PENet.vis_multi_hot(frame, object_poses, true);

            // vwriter.write(pedstimg);
            /*imshow("Detect", dstimg);
            if (cv::waitKey(1) > 0) {
                break;
            };*/

        }else
        {
            // vwriter.write(PENet.vis_multi_hot(frame, object_poses));
        }

    }catch (exception& e)
    {
        cout << e.what() << endl;
        return 0;
    }
    return 1;

}


void find_the_same_obj(std::vector<shared_ptr<byte_track::STrack>>& track_data, std::vector<ObjectPose>& object_poses)
{
    std::vector<std::vector<float>> iouResult(track_data.size(), std::vector<float>(object_poses.size()));
    for(int i = 0; i < track_data.size(); i ++)
    {
        #pragma omp parallel for
        for(int j = 0;j < object_poses.size(); j ++)
        {
            iouResult[i][j] = GetIoU(cv::Rect(track_data[i].get()->getRect().x(), track_data[i].get()->getRect().y(),
                track_data[i].get()->getRect().width(), track_data[i].get()->getRect().height()), object_poses[j].rect);
        }
    }
    auto _track_data = track_data;
    auto _object_poses = object_poses;
    object_poses.clear();
    while (!iouResult.empty())
    {
        float maxVal = -FLT_MAX;
        int maxRow = -1, maxCol = -1;

        for (int i = 0; i < iouResult.size(); ++i) {
            for (int j = 0; j < iouResult[i].size(); ++j) {
                if (iouResult[i][j] > maxVal) {
                    maxVal = iouResult[i][j];
                    maxRow = i;
                    maxCol = j;
                }
            }
        }

        _object_poses[maxCol].track_id = _track_data[maxRow]->getTrackId();
        object_poses.emplace_back(_object_poses[maxCol]);
        iouResult.erase(iouResult.begin() + maxRow);
        _track_data.erase(_track_data.begin() + maxRow);
        _object_poses.erase(_object_poses.begin() + maxCol);
        for (auto& i: iouResult)
        {
            i.erase(i.begin() + maxCol);
        }

    }


}



int detect_one(Mat& frame, TAD& TADnet, DWPOSE& DWPOSENet, byte_track::BYTETracker& BT, VideoWriter& vwriter, int width, int height)
{
    // vector<deep_sort::DetectBox> track_objects;
    vector<ObjectPose> object_poses;
    try{
        // vector<Bbox> boxes;
        // vector<float> det_conf;
        // vector<vector<float>> cls_conf;
        // vector<int> keep_inds = TADNet.detect_one_hot(frame, boxes, det_conf, cls_conf); ////keep_inds记录vector里面的有效检测框的序号
        // Mat dstimg = TADNet.vis_one_hot(frame, boxes, det_conf, cls_conf, keep_inds, vis_thresh);

        auto s_time = std::chrono::system_clock::now();
        TADnet.detect_one_hot(frame, object_poses);
        auto e_time = std::chrono::system_clock::now();
        std::chrono::duration<float> diff = e_time - s_time;
        cout << "YOWOV3net TIME: " << diff.count() << endl;
        vector<byte_track::Object> track_objects;

        if (!object_poses.empty())
        {
            for(auto& o: object_poses)
            {
                if (o.label != 0) continue;
                byte_track::Rect<float> r(o.rect.x, o.rect.y, o.rect.width, o.rect.height);
                byte_track::Object ob(r, o.label, o.prob);
                track_objects.emplace_back(ob);

            }
        }else
        {
            vwriter.write(TADnet.vis(frame, object_poses));
            return 1;
        }
        s_time = std::chrono::system_clock::now();
        auto track_data = BT.update(track_objects);
        e_time = std::chrono::system_clock::now();
        diff = e_time - s_time;
        cout << "TRACK TIME: " << diff.count() << endl;
        if (!track_data.empty())
        {
            find_the_same_obj(track_data, object_poses);
            std::map<unsigned long, Mat> track_imgs;
            for (auto & object_pose : object_poses)
            {
                track_imgs[object_pose.track_id] = frame(object_pose.rect).clone();
            }
            s_time = std::chrono::system_clock::now();
            DWPOSENet.detect(track_imgs, object_poses);
            e_time = std::chrono::system_clock::now();
            diff = e_time - s_time;
            cout << "DWPOSE TIME: " << diff.count() << endl;
        }
        Mat pedstimg = TADnet.vis(frame, object_poses, true);
        vwriter.write(pedstimg);

        /*imshow("Detect", dstimg);
        if (cv::waitKey(1) > 0) {
            break;
        };*/

    }catch (exception& e)
    {
        cout << e.what() << endl;
        return 0;
    }
    return 1;

}


int main(int argc, char* argv[]) {

    const string videopath = R"(/home/linaro/6A/videos/娱越体育运动超人篮球操教学视频.mp4)";
    const string savepath = R"(/home/linaro/6A/videos/result-action-full-娱越体育运动超人篮球操教学视频.mp4)";
    VideoCapture vcapture(videopath);
    if (!vcapture.isOpened())
    {
        cout << "VideoCapture,open video file failed, " << videopath << endl;
        return -1;
    }
    int height = vcapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int width = vcapture.get(cv::CAP_PROP_FRAME_WIDTH);


    TAD TADNet(R"(/home/linaro/6A/model_zoo/yowo_v2_tiny_ava-int8.bmodel)", height, width);
    // PE PENet(R"(/home/linaro/6A/model_zoo/yolov8s-pose-person-face-no-dynamic.bmodel)", 0.6, 0.6);

    // YOWOV3 YOWOV3Net(R"(/home/linaro/6A/model_zoo/yowov3-default.bmodel)");
    DWPOSE DWPOSENet(R"(/home/linaro/6A/model_zoo/dw-mm_ucoco-mod-int8.bmodel)", height, width);

    int fps = vcapture.get(cv::CAP_PROP_FPS);
    int video_length = vcapture.get(cv::CAP_PROP_FRAME_COUNT);

    // deep_sort::DeepSort DS(R"(/home/linaro/6A/model_zoo/osnet.bmodel)", 128, FEATURE_VECTOR_DIM, 0);

    byte_track::BYTETracker tracker(fps, fps * 2, 0.1, 0.2, 0.89);

    VideoWriter vwriter;
    vwriter.open(savepath,
                 cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                 fps,
                 Size(width, height));

    Mat frame;
    int current_frame_id = -1;
    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    std::chrono::duration<float> diff;

    while (vcapture.read(frame))
    {
        current_frame_id += 1;

        if (frame.empty())
        {
            cout << "cv::imread source file failed, " << videopath;
            return -1;
        }
        cout << "frame id :" << current_frame_id << endl;
        cout << endl;
        start_time = std::chrono::system_clock::now();
        int ret = detect_one(frame, TADNet, DWPOSENet, tracker, vwriter, width, height);
        end_time = std::chrono::system_clock::now();
        diff = end_time - start_time;

        cout << "Detect Time: " << diff.count() << endl;

        if(ret == 0)
        {
            cout << "ERROR AT: " << current_frame_id << endl;
            return 0;
        }

        // if (current_frame_id >= 60) break;

    }



    vwriter.release();
    vcapture.release();
    // cout << "向前推理平均值：" << accumulate(begin(TADNet.diffs), end(TADNet.diffs), 0.0) / (float)TADNet.diffs.size() << endl;

    return 0;
}
