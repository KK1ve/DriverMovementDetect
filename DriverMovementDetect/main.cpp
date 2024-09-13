#include "include/TemporalActionDetection.h"
#include "include/PoseEstimation.h"
#include <iostream>
#include <map>
#include <numeric>
// #include "BYTETracker.h"
#include "deepsort.h"
using namespace std;

void detect(Mat& frame, PE& PENet, TAD& TADNet, deep_sort::DeepSort& DS, VideoWriter& vwriter, int width, int height)
{
    vector<deep_sort::DetectBox> track_objects;
    vector<ObjectPose> object_poses;
    try{
        // vector<Bbox> boxes;
        // vector<float> det_conf;
        // vector<vector<float>> cls_conf;
        // vector<int> keep_inds = TADNet.detect_one_hot(frame, boxes, det_conf, cls_conf); ////keep_inds记录vector里面的有效检测框的序号
        // Mat dstimg = TADNet.vis_one_hot(frame, boxes, det_conf, cls_conf, keep_inds, vis_thresh);


        PENet.detect_multi_hot(frame, object_poses);
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
            return;
        }

        DS.sort(frame, track_objects);
        if (!track_objects.empty())
        {
            std::map<unsigned long, Mat> track_imgs;
            for(int i = 0; i < track_objects.size(); i ++)
            {
                if(i < object_poses.size())
                {
                    object_poses[i].track_id = track_objects[i].trackID;
                    track_imgs[object_poses[i].track_id] = frame(object_poses[i].rect).clone();
                }

            }


            auto action_result = TADNet.detect_multi_hot(track_imgs);
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
    }

}


int main(int argc, char* argv[]) {

    const string videopath = R"(/home/lsy/CProject/DriverActionDetect/videos/娱越体育 运动超人篮球操 教学视频.mp4)";
    const string savepath = R"(/home/lsy/CProject/DriverActionDetect/videos/娱越体育 运动超人篮球操 教学视频-result.mp4)";
    VideoCapture vcapture(videopath);
    if (!vcapture.isOpened())
    {
        cout << "VideoCapture,open video file failed, " << videopath << endl;
        return -1;
    }
    int height = vcapture.get(cv::CAP_PROP_FRAME_HEIGHT);
    int width = vcapture.get(cv::CAP_PROP_FRAME_WIDTH);


    TAD TADNet(R"(/home/lsy/CProject/DriverActionDetect/model_zoo/x3d_video.onnx)", height, width);
    PE PENet(R"(/home/lsy/CProject/DriverActionDetect/model_zoo/yolov8s-pose-person-face-no-dynamic.onnx)", 0.6, 0.6);


    int fps = vcapture.get(cv::CAP_PROP_FPS);
    int video_length = vcapture.get(cv::CAP_PROP_FRAME_COUNT);

    deep_sort::DeepSort DS(R"(/home/lsy/CProject/DriverActionDetect/model_zoo/osnet.onnx)", 128, FEATURE_VECTOR_DIM, 0);


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
        end_time = std::chrono::system_clock::now();
        diff = end_time - start_time;
        cout << "总 向前推理时间：" << diff.count() << endl;
        start_time = std::chrono::system_clock::now();
        current_frame_id += 1;

        if (frame.empty())
        {
            cout << "cv::imread source file failed, " << videopath;
            return -1;
        }
        cout << "frame id :" << current_frame_id << endl;

        detect(frame, PENet, TADNet, DS, vwriter, width, height);

    }



    vwriter.release();
    vcapture.release();
    destroyAllWindows();
    // cout << "向前推理平均值：" << accumulate(begin(TADNet.diffs), end(TADNet.diffs), 0.0) / (float)TADNet.diffs.size() << endl;

    return 0;
}
