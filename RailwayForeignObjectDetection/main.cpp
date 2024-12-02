//
// Created by 171153 on 2024/8/13.
//
#include "include/InstanceSegmentation.h"
#include <iostream>
int main(int argc, char* argv[]){
    IS ISNet(R"(/home/linaro/6A/model_zoo/railtrack_segmentation-mix.bmodel)", 0);


    const string videopath = R"(/home/linaro/6A/videos/test-railway-4.mp4)";
    const string savepath = R"(/home/linaro/6A/videos/result-railway-full-4.mp4)";
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
    VideoWriter vwriter;
    vwriter.open(savepath,
                 cv::VideoWriter::fourcc('m', 'p', '4', 'v'),
                 fps,
                 Size(width, height));

    std::chrono::system_clock::time_point start_time = std::chrono::system_clock::now();
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    std::chrono::duration<float> diff;
    try
    {
        Mat frame;
        while (vcapture.read(frame))
        {
            if (frame.empty())
            {
                cout << "cv::imread source file failed, " << videopath;
                break;
            }


            vector<ObjectSeg> object_segs;
            ISNet.detect(frame, object_segs);
            Mat pedstimg = ISNet.vis(frame, object_segs);
            vwriter.write(pedstimg);
        }
    }catch (exception& e)
    {
        cout << e.what() << endl;
    }


    vwriter.release();
    vcapture.release();

    end_time = std::chrono::system_clock::now();
    diff = end_time - start_time;
    start_time = std::chrono::system_clock::now();

    cout << "spend all time: " << diff.count() << endl;
    // cout << "向前推理平均值：" << accumulate(begin(TADNet.diffs), end(TADNet.diffs), 0.0) / (float)TADNet.diffs.size() << endl;

    return 0;
}