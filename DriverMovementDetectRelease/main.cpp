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
#include <tbb/parallel_for.h>

using namespace std;

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


    tbb::flow::source_node<CommonResultPose>
    video_capture(g, [&vcapture](CommonResultPose& input) -> bool
    {
        static unsigned long frame_index = 0;
        static bool is_end = false;
        while (!is_end)
        {
            frame_index += 1;
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

    tbb::flow::multifunction_node<CommonResultPose, std::tuple<CommonResultPose>> sort_video_mat(
    g, 1, [&TADNet](CommonResultPose input, tbb::flow::multifunction_node<CommonResultPose, std::tuple<CommonResultPose>>::output_ports_type& ports)
    {
        static map<unsigned long, Mat> maps;
        static unsigned long current_save_frame_index = 1;
        if (input.frame_index == 0)
        {
            std::get<0>(ports).try_put(input);
            return;
        }
        maps.insert(std::pair<unsigned long, Mat>(input.frame_index, input.origin_mat));
        while (true)
        {
            for(unsigned long i = current_save_frame_index; i < current_save_frame_index + TADNet.len_clip; i ++)
            {
                if(!maps.count(i))
                {
                    return;
                }
            }
            vector<Mat> _video_mats(TADNet.len_clip);
            tbb::parallel_for(0, TADNet.len_clip, [&](int i)
            {
                _video_mats[i] = maps[i + current_save_frame_index];
            });
            maps.erase(current_save_frame_index);
            current_save_frame_index += 1;
            input.video_mats = _video_mats;
            std::get<0>(ports).try_put(input);
        }
    });

    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    pre_process_tad(g, 1, [&TADNet](CommonResultPose input) // CONCURRENCY_EDIT
    {
        if (input.frame_index == 0) return input;
        auto vector_result = TADNet.pre_process(input.video_mats);
        input.video_mats.clear();
        input.float_vector = vector_result;
        return input;
    });

    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    detect_tad(g, 1, [&TADNet](CommonResultPose input)
    {
        if (input.frame_index == 0) return input;
        auto result = TADNet.detect(input);
        cout << "detect size: " << result.object_poses.size() << endl;
        return result;
    });


    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    post_process_tad(g, 1, [&TADNet](CommonResultPose input)    // CONCURRENCY_EDIT
    {
        if (input.frame_index == 0) return input;
        auto result = TADNet.post_process(input);
        return result;
    });

    tbb::flow::multifunction_node<CommonResultPose, std::tuple<CommonResultPose> > sort_tad_result(
    g, 1, [&TADNet](CommonResultPose input, tbb::flow::multifunction_node<CommonResultPose, std::tuple<CommonResultPose> >::output_ports_type& ports)
    {
        static map<unsigned long, CommonResultPose> maps;
        static unsigned long current_save_frame_index = TADNet.len_clip;
        if (input.frame_index == 0)
        {
            std::get<0>(ports).try_put(input);
            return;
        }
        vector<byte_track::Object> track_objects;
        for(auto& o: input.object_poses)
        {
            byte_track::Rect<float> r(o.rect.x, o.rect.y, o.rect.width, o.rect.height);
            byte_track::Object ob(r, o.label, o.prob, o.action_prob);
            track_objects.emplace_back(ob);
        }
        input.track_objects = track_objects;

        maps.insert(std::pair<unsigned long, CommonResultPose>(input.frame_index, input));

        while (maps.count(current_save_frame_index))
        {
            CommonResultPose result(maps[current_save_frame_index]);
            std::get<0>(ports).try_put(result);
            maps.erase(current_save_frame_index);
            current_save_frame_index += 1;
        }
    });


    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    track(g, 1, [&tracker](CommonResultPose input)
    {
        auto track_result = tracker.update(input.track_objects);
        input.track_vector = track_result;
        input.track_objects.clear();
        return input;
    });

    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    pre_process_pose(g, 1, [&DWPOSENet](CommonResultPose input) // CONCURRENCY_EDIT
    {
        if (input.frame_index == 0) return input;
        cout << "track size: " << input.track_vector.size() << endl;
        return DWPOSENet.pre_process(input);
    });


    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    detect_pose(g, 1, [&DWPOSENet](CommonResultPose input)  // CONCURRENCY_EDIT
    {
        if (input.frame_index == 0) return input;
        return DWPOSENet.detect(input);
    });


    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    post_process_pose(g, 1, [&DWPOSENet](CommonResultPose input)    // CONCURRENCY_EDIT
    {
        if (input.frame_index == 0) return input;
        return DWPOSENet.post_process(input);
    });

    tbb::flow::function_node<CommonResultPose, CommonResultPose>
    vis(g, 1, [&TADNet](CommonResultPose input) // CONCURRENCY_EDIT
    {
        if (input.frame_index == 0) return input;
        return TADNet.vis(input);
    });
    tbb::flow::queue_node<CommonResultPose> queue_node(g);
    tbb::flow::broadcast_node<CommonResultPose> broadcast_node(g);


    tbb::flow::function_node<CommonResultPose>
    save(g, 1, [&vwriter](CommonResultPose input)
    {
        // cout << "save: " << input.frame_index << endl;
        static map<unsigned long, cv::Mat> hash_map;
        static unsigned long current_save_frame_index = 1;
        if (input.frame_index == 0) return;
        cv::Mat mat(input.processed_mat);
        const unsigned long _frame_index = input.frame_index;
        hash_map.insert(std::pair<unsigned long, cv::Mat>(_frame_index, mat));
        while (hash_map.count(current_save_frame_index))
        {
            cout << "write video: " << current_save_frame_index << endl;
            vwriter.write(hash_map[current_save_frame_index]);
            hash_map.erase(current_save_frame_index);
            current_save_frame_index += 1;
        }
        // cout << "save done: " << input.frame_index << endl;
    });

    tbb::flow::make_edge(video_capture, sort_video_mat);
    tbb::flow::make_edge(tbb::flow::output_port<0>(sort_video_mat), pre_process_tad);
    tbb::flow::make_edge(pre_process_tad, detect_tad);
    tbb::flow::make_edge(detect_tad, post_process_tad);
    tbb::flow::make_edge(post_process_tad, sort_tad_result);
    tbb::flow::make_edge(tbb::flow::output_port<0>(sort_tad_result), track);
    tbb::flow::make_edge(track, pre_process_pose);
    tbb::flow::make_edge(pre_process_pose, detect_pose);
    tbb::flow::make_edge(detect_pose, post_process_pose);
    tbb::flow::make_edge(post_process_pose, vis);
    tbb::flow::make_edge(vis, queue_node);
    tbb::flow::make_edge(queue_node, broadcast_node);
    // tbb::flow::make_edge(broadcast_node, /*your code here*/);
    tbb::flow::make_edge(broadcast_node, save);

    video_capture.activate();
    g.wait_for_all();



    vwriter.release();
    vcapture.release();
    // cout << "向前推理平均值：" << accumulate(begin(TADNet.diffs), end(TADNet.diffs), 0.0) / (float)TADNet.diffs.size() << endl;

    return 0;
}
