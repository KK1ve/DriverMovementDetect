//
// Created by 171153 on 2024/8/13.
//
#include "include/InstanceSegmentation.h"
#include <iostream>
int main(int argc, char* argv[]){


    IS ISNet(R"(/home/lsy/CProject/DriverActionDetect/model_zoo/railtrack_segmentation.onnx)");

    const string videopath = R"(/home/lsy/CProject/DriverActionDetect/test-railway-4.mp4)";
    const string savepath = R"(/home/lsy/CProject/DriverActionDetect/result-railway-4.mp4)";
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
    try
    {
        Mat frame;
        while (vcapture.read(frame))
        {
            if (frame.empty())
            {
                cout << "cv::imread source file failed, " << videopath;
                return -1;
            }

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
    destroyAllWindows();
    // cout << "向前推理平均值：" << accumulate(begin(TADNet.diffs), end(TADNet.diffs), 0.0) / (float)TADNet.diffs.size() << endl;

    return 0;
}