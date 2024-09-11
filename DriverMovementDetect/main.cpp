#include "include/TemporalActionDetection.h"
#include "include/PoseEstimation.h"
#include <iostream>
#include <numeric>
#include "BYTETracker.h"
using namespace std;



int main(int argc, char* argv[]) {
    const float vis_thresh = 0.3;

    const string videopath = R"(/home/lsy/CProject/DriverActionDetect/拳击  实战  对抗.mp4)";
    const string savepath = R"(/home/lsy/CProject/DriverActionDetect/拳击  实战  对抗-result.mp4)";
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

    byte_track::BYTETracker tracker(fps, fps * 2, 0.5, 0.75, 0.9);


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
            // vector<int> keep_inds = TADNet.detect_one_hot(frame, boxes, det_conf, cls_conf); ////keep_inds记录vector里面的有效检测框的序号
            // Mat dstimg = TADNet.vis_one_hot(frame, boxes, det_conf, cls_conf, keep_inds, vis_thresh);


            vector<ObjectPose> object_poses;
            PENet.detect_multi_hot(frame, object_poses);

            vector<byte_track::Object> track_objects;
            if (!object_poses.empty())
            {
                for(auto& o: object_poses)
                {
                    if(o.label == 0)
                    {
                        byte_track::Rect<float> r(o.rect.x, o.rect.y, o.rect.width, o.rect.height);
                        byte_track::Object ob(r, o.label, o.prob);
                        track_objects.emplace_back(ob);
                    }

                }
            }else
            {
                vwriter.write(PENet.vis_multi_hot(frame, object_poses));
                continue;
            }

            if (!track_objects.empty())
            {
                auto track_data = tracker.update(track_objects);
                if (track_data.empty())
                {
                    vwriter.write(PENet.vis_multi_hot(frame, object_poses));
                    continue;
                }

                std::map<unsigned long, Mat> track_imgs;

                for (int i = 0; i < track_data.size(); i ++)
                {
                    int _x = track_data[i]->getRect().x() > width ? width : track_data[i]->getRect().x() < 0 ? 0 : track_data[i]->getRect().x();
                    int _y = track_data[i]->getRect().y() > height ? height : track_data[i]->getRect().y() < 0 ? 0 : track_data[i]->getRect().y();
                    int _width = track_data[i]->getRect().width() > width - _x ? width - _x : track_data[i]->getRect().width();
                    int _height = track_data[i]->getRect().height() > height - _y ? height - _y : track_data[i]->getRect().height();

                    object_poses[i].track_id = track_data[i]->getTrackId();
                    track_imgs[track_data[i]->getTrackId()] = frame(cv::Rect(
                        _x,
                        _y,
                        _width,
                        _height));
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
