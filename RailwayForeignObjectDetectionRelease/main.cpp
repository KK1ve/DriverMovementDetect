//
// Created by 171153 on 2024/8/13.
//
#include <atomic>

#include "include/InstanceSegmentation.h"
#include <condition_variable>
#include <utils.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/flow_graph.h>
#include <map>

std::atomic<unsigned int> condition{1};
std::mutex mtx;
std::condition_variable my_variable;

int main(int argc, char* argv[]){
    IS ISNet(R"(/home/linaro/6A/model_zoo/railtrack_segmentation-mod-mix.bmodel)");

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
    condition = fps;
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

    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
    std::chrono::duration<float> diff;

    tbb::flow::source_node<CommonResultSeg>
    video_capture(g, [&vcapture](CommonResultSeg& input) -> bool
    {
        static unsigned long frame_index = 0;
        static bool is_end = false;
        while (!is_end)
        {
            // cout << "video capture: " << frame_index <<endl;
            frame_index += 1;
            Mat video_mat;
            std::unique_lock<std::mutex> lock(mtx);
            my_variable.wait(lock, [] { return condition.load() > 0; });
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
            input.start_time = std::chrono::system_clock::now(); // TODO CAN BE DELETE
            // cout << "video capture done: " << frame_index << endl;
            return true;
        }
        return false;

    }, false);

    tbb::flow::limiter_node<CommonResultSeg> limiter(g, 1);


    tbb::flow::function_node<CommonResultSeg, CommonResultSeg>
    pre_process(g, 0, [&ISNet](CommonResultSeg input)
    {
        input.start_time = std::chrono::system_clock::now(); // TODO CAN BE DELETE
        if (input.frame_index == 0) return input;
        auto result = ISNet.pre_process(input);
        std::chrono::duration<float> _diff = std::chrono::system_clock::now() - input.start_time; // TODO CAN BE DELETE
        cout << "pre_process done: " << input.frame_index << " use time: " <<  _diff.count() << endl; // TODO CAN BE DELETE
        return result;
    });


    tbb::flow::function_node<CommonResultSeg, CommonResultSeg>
    detect(g, 1, [&ISNet](CommonResultSeg input)
    {
        input.start_time = std::chrono::system_clock::now(); // TODO CAN BE DELETE
        if (input.frame_index == 0) return input;
        auto result = ISNet.detect(input);
        std::chrono::duration<float> _diff = std::chrono::system_clock::now() - input.start_time; // TODO CAN BE DELETE
        std::lock_guard<std::mutex> lock(mtx);
        condition += 1;
        cout << "detect done: " << input.frame_index << " use time: " <<  _diff.count() << endl; // TODO CAN BE DELETE
        return result;
    });


    tbb::flow::function_node<CommonResultSeg, CommonResultSeg>
    post_process(g, 0, [&ISNet](CommonResultSeg input)
    {
        input.start_time = std::chrono::system_clock::now(); // TODO CAN BE DELETE
        if (input.frame_index == 0) return input;
        auto result = ISNet.post_process(input);
        std::chrono::duration<float> _diff = std::chrono::system_clock::now() - input.start_time; // TODO CAN BE DELETE
        cout << "post_process done: " << input.frame_index << " use time: " <<  _diff.count() << endl; // TODO CAN BE DELETE
        return result;
    });


    tbb::flow::function_node<CommonResultSeg, CommonResultSeg>
    vis(g, 0, [&ISNet, &diff, &start_time](CommonResultSeg input)
    {
        input.start_time = std::chrono::system_clock::now(); // TODO CAN BE DELETE
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
        //     cout << "frame index: " << input.frame_index << "  use time: " << _diff.count() << endl;
        // }
        std::chrono::duration<float> _diff = std::chrono::system_clock::now() - input.start_time; // TODO CAN BE DELETE
        cout << "vis done: " << input.frame_index << " use time: " <<  _diff.count() << endl; // TODO CAN BE DELETE
        return vis_result;
    });
    tbb::flow::queue_node<CommonResultSeg> queue_node(g);
    tbb::flow::broadcast_node<CommonResultSeg> broadcast_node(g);

    tbb::flow::function_node<CommonResultSeg>
    save(g, 1, [&vwriter](CommonResultSeg input)
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

    // tbb::flow::function_node<CommonResultSeg>
    // save_batch(g, 0, [&vwriter, &max_frame_id](CommonResultSeg input)
    // {
    //     if (input.frame_index == 0) return;
    //     std::unique_lock<std::mutex> lock(video_write_variable_mtx);
    //     video_write_variable.wait(lock, [&input, &max_frame_id]{return input.frame_index == max_frame_id + 1;});
    //     static map<unsigned long, CommonResultSeg> mat_map;
    //     mat_map.insert(std::pair<unsigned long, CommonResultSeg>(input.frame_index,input));
    //     // TODO
    // });

    tbb::flow::make_edge(video_capture, pre_process);
    tbb::flow::make_edge(pre_process, detect);
    tbb::flow::make_edge(detect, post_process);
    tbb::flow::make_edge(post_process, vis);
    tbb::flow::make_edge(vis, queue_node);
    tbb::flow::make_edge(queue_node, broadcast_node);
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