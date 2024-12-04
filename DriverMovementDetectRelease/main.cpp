#include "include/TemporalActionDetection.h"
#include "include/DWPOSE.h"
#include <iostream>
#include <map>
#include <numeric>
#include "deepsort.h"
#include "BYTETracker.h"
#include <condition_variable>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/flow_graph.h>


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
        cout << "TAD ACTION PRED TIME: " << diff.count() << endl;
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
            std::vector<cv::Mat> track_imgs;
            for (auto & object_pose : object_poses)
            {
                track_imgs.emplace_back(frame(object_pose.rect).clone());
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

condition_variable video_capture_variable;
std::mutex video_capture_variable_mtx;

condition_variable pre_process_variable;
std::mutex pre_process_variable_mtx;

std::mutex need_capture_mtx;

condition_variable video_write_variable;
std::mutex video_write_variable_mtx;

std::mutex video_write_mtx;

std::mutex vcapture_mats_mtx;

condition_variable max_frame_id_track_variable;
std::mutex max_frame_id_track_mtx;

int main(int argc, char*  []) {

    const string videopath = R"(/home/linaro/6A/videos/火车司机ppt检测.mp4)";
    const string savepath = R"(/home/linaro/6A/videos/result-full-火车司机ppt检测.mp4)";
    VideoCapture vcapture(videopath);
    if (!vcapture.isOpened())
    {
        cout << "VideoCapture,open video file failed, " << videopath << endl;
        return -1;
    }
    int height = vcapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int width = vcapture.get(cv::CAP_PROP_FRAME_WIDTH);
    int fps = vcapture.get(cv::CAP_PROP_FPS);
    int video_length = vcapture.get(cv::CAP_PROP_FRAME_COUNT);

    TAD TADNet(R"(/home/linaro/6A/model_zoo/yowo_v2_tiny_ava-mix.bmodel)", height, width);
    DWPOSE DWPOSENet(R"(/home/linaro/6A/model_zoo/dw-mm_ucoco-mod-mix.bmodel)", height, width);

    byte_track::BYTETracker tracker(fps, fps * 2, 0.1, 0.2, 0.89);

    VideoWriter vwriter;
    vwriter.open(savepath,
                 cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                 fps,
                 Size(width, height));

    tbb::flow::graph g;

    unsigned long need_capture = 1;
    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
    std::chrono::duration<float> diff;

    tbb::flow::source_node<CommonResultPose>
    video_capture(g, [&vcapture, &need_capture](CommonResultPose& input) -> bool
    {
        static unsigned long frame_index = 0;
        static bool is_end = false;
        while (!is_end)
        {
            frame_index += 1;
            {
                std::unique_lock<std::mutex> video_capture_lock(video_capture_variable_mtx);
                video_capture_variable.wait(video_capture_lock, [&need_capture]{return need_capture > 0;});
            }
            {
                std::unique_lock<std::mutex> need_capture_lock(need_capture_mtx);
                need_capture -= 1;
            }
            Mat video_mat;
            vcapture.read(video_mat);
            if (video_mat.empty())
            {
                is_end = true;
                input.frame_index = 0;
                input.origin_mat = Mat();
                // model_input_mat.start_time = std::chrono::system_clock::now(); // TODO CAN BE DELETE
                return true;
            }
            input.frame_index = frame_index;
            input.origin_mat = video_mat;
            // model_input_mat.start_time = std::chrono::system_clock::now(); // TODO CAN BE DELETE
            return true;
        }
        return false;

    }, false);

    unsigned long pre_process_max_frame_id = 0;
    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    pre_process_tad(g, 0, [&TADNet, &pre_process_max_frame_id](CommonResultPose input)
    {
        if (input.frame_index == 0) return input;
        static std::vector<Mat> vcapture_mats;
        std::unique_lock<std::mutex> lock(pre_process_variable_mtx);
        pre_process_variable.wait(lock, [&input, &pre_process_max_frame_id]{return input.frame_index == pre_process_max_frame_id + 1;});
        vector<Mat> temp_vcapture_mats;
        vector<float> vector_result;
        Mat precessed_mat = TADNet.pre_process_mat(input.origin_mat);
        {
            std::unique_lock<std::mutex> vcapture_mats_lock(vcapture_mats_mtx);
            vcapture_mats.emplace_back(precessed_mat);
            if(vcapture_mats.size() > TADNet.len_clip)
            {
                vcapture_mats.erase(vcapture_mats.begin());
            }else if(vcapture_mats.size() <= TADNet.len_clip)
            {
                for(int i = 0; i < TADNet.len_clip - vcapture_mats.size(); i ++)
                {
                    vcapture_mats.emplace_back(precessed_mat);
                }
            }
            temp_vcapture_mats = vcapture_mats;
            pre_process_max_frame_id += 1;
        }
        vector_result = TADNet.pre_process(temp_vcapture_mats);
        CommonResultPose result(input);
        result.float_vector = vector_result;
        return result;
    });

    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    detect_tad(g, 2, [&TADNet, &need_capture](CommonResultPose input)
    {
        if (input.frame_index == 0) return input;
        auto result = TADNet.detect(input);
        {
            std::unique_lock<std::mutex> lock(need_capture_mtx);
            need_capture += 1;
        }
        video_capture_variable.notify_all();
        return result;
    });


    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    post_process_tad(g, 0, [&TADNet](CommonResultPose input)
    {
        if (input.frame_index == 0) return input;
        auto result = TADNet.post_process(input);
        return result;
    });

    unsigned long max_frame_id_track = 0;

    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    track(g, 0, [&tracker, &max_frame_id_track](CommonResultPose input)
    {
        if (input.frame_index == 0) return input;
        vector<byte_track::Object> track_objects;
        for(auto& o: input.object_poses)
        {
            byte_track::Rect<float> r(o.rect.x, o.rect.y, o.rect.width, o.rect.height);
            byte_track::Object ob(r, o.label, o.prob, o.action_prob);
            track_objects.emplace_back(ob);
        }
        std::unique_lock<std::mutex> lock(max_frame_id_track_mtx);
        max_frame_id_track_variable.wait(lock, [&input, &max_frame_id_track]{return input.frame_index == max_frame_id_track + 1;});
        auto track_result = tracker.update(track_objects);
        CommonResultPose result(input);
        result.track_vector = track_result;
        max_frame_id_track += 1;
        return result;
    });

    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    pre_process_pose(g, 0, [&DWPOSENet](CommonResultPose input)
    {
        if (input.frame_index == 0) return input;
        return DWPOSENet.pre_process(input);
    });


    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    detect_pose(g, 0, [&DWPOSENet](CommonResultPose input)
    {
        if (input.frame_index == 0) return input;
        return DWPOSENet.detect(input);
    });


    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    post_process_pose(g, 0, [&DWPOSENet](CommonResultPose input)
    {
        if (input.frame_index == 0) return input;
        return DWPOSENet.post_process(input);
    });

    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    vis(g, 0, [&TADNet](CommonResultPose input)
    {
        if (input.frame_index == 0) return input;
        return TADNet.vis(input);
    });

    tbb::flow::broadcast_node<CommonResultSeg> broadcast_node(g);

    unsigned long max_frame_id = 0;

    tbb::flow::function_node<CommonResultSeg>
    save(g, 0, [&vwriter, &max_frame_id](CommonResultSeg input)
    {
        if (input.frame_index == 0) return;
        std::unique_lock<std::mutex> lock(video_write_variable_mtx);
        video_write_variable.wait(lock, [&input, &max_frame_id]{return input.frame_index == max_frame_id + 1;});
        {
            cout << "write video: " << input.frame_index << endl;
            std::unique_lock<std::mutex> video_write_lock(video_write_mtx);
            vwriter.write(input.processed_mat);
            max_frame_id += 1;
        }
        video_write_variable.notify_all();
    });







    vwriter.release();
    vcapture.release();
    // cout << "向前推理平均值：" << accumulate(begin(TADNet.diffs), end(TADNet.diffs), 0.0) / (float)TADNet.diffs.size() << endl;

    return 0;
}
