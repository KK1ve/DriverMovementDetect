//
// Created by 171153 on 2024/8/13.
//
#include "include/InstanceSegmentation.h"
#include <iostream>
int main(int argc, char* argv[]){
    IS ISNet(R"(/home/linaro/6A/model_zoo/railtrack_segmentation-int8.bmodel)", 0);


    const string videopath = R"(/home/linaro/6A/videos/test-railway-4.mp4)";
    const string savepath = R"(/home/linaro/6A/videos/result-railway-4.mp4)";
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
        int lenas = 40;
        Mat frame;
        while (vcapture.read(frame))
        {
            if (frame.empty())
            {
                cout << "cv::imread source file failed, " << videopath;
                return -1;
            }
            if (lenas -- < 0)
            {
                break;
            }

            end_time = std::chrono::system_clock::now();
            diff = end_time - start_time;
            start_time = std::chrono::system_clock::now();

            cout << "Time: " << diff.count() << endl;

            // vector<Bbox> boxes;
            // vector<float> det_conf;
            // vector<vector<float>> cls_conf;
            // vector<int> keep_inds = TADNet.detect_multi_hot(frame, boxes, det_conf, cls_conf); ////keep_inds记录vector里面的有效检测框的序号
            // Mat dstimg = TADNet.vis_multi_hot(frame, boxes, det_conf, cls_conf, keep_inds, vis_thresh);


            vector<ObjectSeg> object_segs;
            ISNet.detect(frame, object_segs);
            Mat pedstimg = ISNet.vis(frame, object_segs);


            vwriter.write(pedstimg);
            /*imshow("Detect", dstimg);
            if (cv::waitKey(1) > 0) {
                break;
            };*/

        }
    }catch (exception& e)
    {
        cout << e.what() << endl;
    }


    vwriter.release();
    vcapture.release();
    // cout << "向前推理平均值：" << accumulate(begin(TADNet.diffs), end(TADNet.diffs), 0.0) / (float)TADNet.diffs.size() << endl;

    return 0;
}