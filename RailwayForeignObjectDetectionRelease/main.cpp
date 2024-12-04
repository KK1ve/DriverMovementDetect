//
// Created by 171153 on 2024/8/13.
//
#include "include/InstanceSegmentation.h"
#include <condition_variable>
#include <utils.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/flow_graph.h>
#include <tbb/concurrent_queue.h>

condition_variable video_capture_variable;
std::mutex video_capture_variable_mtx;

std::mutex need_capture_mtx;

condition_variable video_write_variable;
std::mutex video_write_variable_mtx;

std::mutex video_write_mtx;



int main(int argc, char* argv[]){
    IS ISNet(R"(/home/linaro/6A/model_zoo/railtrack_segmentation-mix.bmodel)", 0);

    const string videopath = R"(/home/linaro/6A/videos/test-railway-4.mp4)";
    const string savepath = R"(/home/linaro/6A/videos/result-railway-mix-test-4.mp4)";
    VideoCapture vcapture(videopath);
    if (!vcapture.isOpened())
    {
        cout << "VideoCapture,open video file failed, " << videopath << endl;
        return -1;
    }
    int height = static_cast<int>(vcapture.get(cv::CAP_PROP_FRAME_HEIGHT));
    int width = static_cast<int>(vcapture.get(cv::CAP_PROP_FRAME_WIDTH));
    int fps = static_cast<int>(vcapture.get(cv::CAP_PROP_FPS));
    int video_length = static_cast<int>(vcapture.get(cv::CAP_PROP_FRAME_COUNT));

    VideoWriter vwriter;
    vwriter.open(savepath,
                 cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                 fps,
                 Size(width, height));

    // VideoWriter testvwriter;
    // auto testvwriteerpath = savepath + ".mp4";
    // testvwriter.open(testvwriteerpath,
    //              cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
    //              fps,
    //              Size(width, height));


    tbb::flow::graph g;

    unsigned long need_capture = 1;
    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
    std::chrono::duration<float> diff;

    tbb::flow::source_node<CommonResultSeg>
    video_capture(g, [&vcapture, &need_capture](CommonResultSeg& input) -> bool
    {
        static unsigned long frame_index = 0;
        static bool is_end = false;
        while (!is_end)
        {
            cout << "video capture: " << frame_index <<endl;
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
                return true;
            }
            input.frame_index = frame_index;
            input.origin_mat = video_mat;
            // model_input_mat.start_time = std::chrono::system_clock::now(); // TODO CAN BE DELETE
            cout << "video capture done: " << frame_index << endl;
            return true;
        }
        return false;

    }, false);


    tbb::flow::function_node<CommonResultSeg, CommonResultSeg>
    pre_process(g, 0, [&ISNet](CommonResultSeg input)
    {
        cout << "pre process: " << input.frame_index << endl;
        if (input.frame_index == 0) return input;
        auto result = ISNet.pre_process(input);
        cout << "pre process done: " << input.frame_index << endl;
        return result;
    });


    tbb::flow::function_node<CommonResultSeg, CommonResultSeg>
    detect(g, 1, [&ISNet, &need_capture](CommonResultSeg input)
    {
        cout << "detect: " << input.frame_index << endl;
        if (input.frame_index == 0) return input;
        auto result = ISNet.detect(input);
        {
            std::unique_lock<std::mutex> lock(need_capture_mtx);
            need_capture += 1;
        }
        video_capture_variable.notify_all();
        cout << "detect done: " << input.frame_index << endl;
        return result;
    });


    tbb::flow::function_node<CommonResultSeg, CommonResultSeg>
    post_process(g, 0, [&ISNet](CommonResultSeg input)
    {
        cout << "post process: " << input.frame_index << endl;
        if (input.frame_index == 0) return input;
        auto result = ISNet.post_process(input);
        cout << "post process done: " << input.frame_index << endl;
        return result;
    });


    tbb::flow::function_node<CommonResultSeg, CommonResultSeg>
    vis(g, 0, [&ISNet, &diff, &start_time](CommonResultSeg input)
    {
        cout << "vis: " << input.frame_index << endl;
        if (input.frame_index == 0)
        {
            diff = std::chrono::system_clock::now() - start_time;
            cout << "inference use time: " << diff.count() << endl;
            return input;
        }
        auto vis_result = ISNet.vis(input);

        // { // TODO CAN BE DELETE
        //
        //     std::chrono::duration<float> _diff = std::chrono::system_clock::now() - input.start_time;
        //     cout << "frame index: " << input.frame_index << "  use time: " << _diff.cout() << endl;
        // }
        cout << "vis done: " << input.frame_index << endl;
        return vis_result;
    });

    tbb::flow::broadcast_node<CommonResultSeg> broadcast_node(g);

    unsigned long max_frame_id = 0;
    tbb::flow::function_node<CommonResultSeg>
    save(g, 0, [&vwriter, &max_frame_id](CommonResultSeg input)
    {
        cout << "save: " << input.frame_index << endl;
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
        cout << "save done: " << input.frame_index << endl;
    });

    tbb::flow::function_node<CommonResultSeg>
    save_batch(g, 0, [&vwriter, &max_frame_id](CommonResultSeg input)
    {
        if (input.frame_index == 0) return;
        std::unique_lock<std::mutex> lock(video_write_variable_mtx);
        video_write_variable.wait(lock, [&input, &max_frame_id]{return input.frame_index == max_frame_id + 1;});
        static map<unsigned long, CommonResultSeg> mat_map;
        mat_map.insert(std::pair<unsigned long, CommonResultSeg>(input.frame_index,input));
        // TODO
    });

    tbb::flow::make_edge(video_capture, pre_process);
    tbb::flow::make_edge(pre_process, detect);
    tbb::flow::make_edge(detect, post_process);
    tbb::flow::make_edge(post_process, vis);
    tbb::flow::make_edge(vis, broadcast_node);
    // tbb::flow::make_edge(broadcast_node, /*your code here*/);
    tbb::flow::make_edge(broadcast_node, save);

    video_capture.activate();

    g.wait_for_all();

    vwriter.release();
    vcapture.release();
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    diff = end_time - start_time;
    cout << "spend all time: " << diff.count() << endl;

    return 0;
}