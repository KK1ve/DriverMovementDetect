//
// Created by 171153 on 2024/8/13.
//
#include "include/InstanceSegmentation.h"
#include <condition_variable>
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

    tbb::flow::source_node<std::tuple<unsigned long, Mat>>
    video_capture(g, [&vcapture, &need_capture](std::tuple<unsigned long, Mat>& model_input_mat) -> bool
    {
        static unsigned long frame_index = 0;
        while (true)
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
                return false;
            }
            std::get<0>(model_input_mat) = frame_index;
            std::get<1>(model_input_mat) = video_mat;
            return true;
        }

    }, false);


    tbb::flow::function_node<std::tuple<unsigned long, Mat>,
    std::tuple<unsigned long, vector<float>, Mat>>
    pre_process(g, 0, [&ISNet](std::tuple<unsigned long, Mat> model_input_mat)
    {
        auto result = ISNet.pre_process(model_input_mat);
        return result;
    });


    tbb::flow::function_node<std::tuple<unsigned long, vector<float>, Mat>,
    std::tuple<unsigned long, std::shared_ptr<BMNNTensor>, Mat>>
    detect(g, 2, [&ISNet, &need_capture](std::tuple<unsigned long, std::vector<float>, Mat> input_vector)
    {
        auto result = ISNet.detect(input_vector);
        {
            std::lock_guard<std::mutex> lock(need_capture_mtx);
            need_capture += 1;
        }
        video_capture_variable.notify_all();
        return result;
    });


    tbb::flow::function_node<std::tuple<unsigned long, std::shared_ptr<BMNNTensor>, Mat>,
    std::tuple<unsigned long, vector<ObjectSeg>, Mat>>
    post_process(g, 0, [&ISNet](std::tuple<unsigned long, std::shared_ptr<BMNNTensor>, Mat> pred)
    {
        auto result = ISNet.post_process(pred);
        return result;
    });


    tbb::flow::function_node<std::tuple<unsigned long, vector<ObjectSeg>, Mat>,
    std::tuple<unsigned long, Mat, Mat>>
    vis(g, 3, [&ISNet](std::tuple<unsigned long, vector<ObjectSeg>, Mat> boxes)
    {
        auto vis_result = ISNet.vis(boxes);
        return vis_result;
    });

    unsigned long max_frame_id = 0;
    tbb::concurrent_vector<std::tuple<unsigned long, Mat, Mat>> concurrent_vector;

    tbb::flow::function_node<std::tuple<unsigned long, Mat, Mat>>
    save(g, 0, [&vwriter, &max_frame_id](std::tuple<unsigned long, Mat, Mat> vis_result)
    {
        std::unique_lock<std::mutex> lock(video_write_variable_mtx);
        while (std::get<0>(vis_result) != max_frame_id + 1)
        {
            video_write_variable.wait(lock);
        }
        {
            cout << "write video: " << std::get<0>(vis_result) << endl;
            std::unique_lock<std::mutex> video_write_lock(video_write_mtx);
            vwriter.write(std::get<1>(vis_result));
            max_frame_id += 1;
        }
        video_write_variable.notify_all();
    });

    tbb::flow::make_edge(video_capture, pre_process);
    tbb::flow::make_edge(pre_process, detect);
    tbb::flow::make_edge(detect, post_process);
    tbb::flow::make_edge(post_process, vis);
    tbb::flow::make_edge(vis, save);

    video_capture.activate();

    g.wait_for_all();

    vwriter.release();
    vcapture.release();
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    diff = end_time - start_time;
    cout << "spend all time: " << diff.count() << endl;

    return 0;
}