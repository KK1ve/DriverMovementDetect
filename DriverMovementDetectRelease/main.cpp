#include "include/TemporalActionDetection.h"
#include "include/DWPOSE.h"
#include <iostream>
#include <map>
#include <numeric>
#include "BYTETracker.h"
#include <condition_variable>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/flow_graph.h>


namespace byte_track
{
    struct Object;
}

using namespace std;

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

    TAD TADNet(R"(/home/linaro/6A/model_zoo/yowo_v2_medium_ava-mix.bmodel)", height, width);
    DWPOSE DWPOSENet(R"(/home/linaro/6A/model_zoo/dw-mm_ucoco-mod-mix.bmodel)", height, width);

    byte_track::BYTETracker tracker(fps, fps * 2, 0.1, 0.2, 0.89);

    VideoWriter vwriter;
    vwriter.open(savepath,
                 cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                 fps,
                 Size(width, height));

    tbb::flow::graph g;

    unsigned long need_capture = 1;

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


    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    pre_process_tad(g, 0, [&TADNet](CommonResultPose input)
    {
        if (input.frame_index == 0) return input;
        static unsigned long pre_process_max_frame_id = 0;
        static std::vector<Mat> vcapture_mats;
        std::unique_lock<std::mutex> lock(pre_process_variable_mtx);
        pre_process_variable.wait(lock, [&input, &pre_process_max_frame_id]{return input.frame_index == pre_process_max_frame_id + 1;});
        vector<Mat> temp_vcapture_mats;
        vector<float> vector_result;
        Mat precessed_mat = TADNet.pre_process_mat(input.origin_mat);
        {
            std::unique_lock<std::mutex> vcapture_mats_lock(vcapture_mats_mtx);
            vcapture_mats.emplace_back(precessed_mat);
            if(vcapture_mats.size() > TADNet.len_clip - 1)
            {
                vcapture_mats.erase(vcapture_mats.begin());
            }else if(vcapture_mats.size() <= TADNet.len_clip)
            {
                const int count = static_cast<int>(TADNet.len_clip - vcapture_mats.size());
                for(int i = 0; i < count; i ++)
                {
                    vcapture_mats.emplace_back(precessed_mat.clone());
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



    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    track(g, 0, [&tracker](CommonResultPose input)
    {
        if (input.frame_index == 0) return input;
        static unsigned long max_frame_id_track = 0;
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

    tbb::flow::broadcast_node<CommonResultPose> broadcast_node(g);


    tbb::flow::function_node<CommonResultPose>
    save(g, 0, [&vwriter](CommonResultPose input)
    {
        if (input.frame_index == 0) return;
        static unsigned long max_frame_id = 0;
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

    tbb::flow::make_edge(video_capture, pre_process_tad);
    tbb::flow::make_edge(pre_process_tad, detect_tad);
    tbb::flow::make_edge(detect_tad, post_process_tad);
    tbb::flow::make_edge(post_process_tad, track);
    tbb::flow::make_edge(track, pre_process_pose);
    tbb::flow::make_edge(pre_process_pose, detect_pose);
    tbb::flow::make_edge(detect_pose, post_process_pose);
    tbb::flow::make_edge(post_process_pose, vis);
    tbb::flow::make_edge(vis, broadcast_node);
    // tbb::flow::make_edge(broadcast_node, /*your code here*/);
    tbb::flow::make_edge(broadcast_node, save);

    video_capture.activate();
    g.wait_for_all();



    vwriter.release();
    vcapture.release();
    // cout << "向前推理平均值：" << accumulate(begin(TADNet.diffs), end(TADNet.diffs), 0.0) / (float)TADNet.diffs.size() << endl;

    return 0;
}
