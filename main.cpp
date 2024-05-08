
#include "layer.h"
#include "net.h"


#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <stdio.h>
#include <vector>
#include <net.h>
#include "yolo-pose.h"
#include "yolov5.h"
#include "yoloface.h"
#include <string>
#include "pose-classification.h"


void draw_dps(cv::Mat frame, std::chrono::system_clock::time_point start_time) {
    char fps[100];
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    sprintf(fps, "FPS: %.1f", (float)1 / (diff.count()));
    cv::putText(frame, fps, cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.75, { 255, 255, 255 }, 1);
}


int main(int argc, char** argv)
{
    int height = 640;
    int width = 480;
    cv::VideoCapture capture;
    capture.open(0);  //修改这个参数可以选择打开想要用的摄像头

    //static Yolo_pose* yoloPose = new Yolo_pose();
    //const float mean_vals[] = { 0.0f, 0.0f, 0.0f };
    //const float norm_vals[] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    //yoloPose->load("yolov8s-face-pose-kk", 416, mean_vals, norm_vals, true);

    //static pfld_pose* pfldPose = new pfld_pose();
    //const float mean_vals[] = { 0.0f, 0.0f, 0.0f };
    //const float norm_vals[] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    //pfldPose->load("pfld-sim", 416, mean_vals, norm_vals, true);

    //static Yolov8* yolov8 = new Yolov8();
    //const float mean_vals[] = { 0.0f, 0.0f, 0.0f };
    //const float norm_vals[] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    //yolov8->load("yolov8-eye-mouth", 416, mean_vals, norm_vals, true);


    static Yolov5* yolov5 = new Yolov5();
    const float mean_vals[] = { 0.0f, 0.0f, 0.0f };
    const float norm_vals[] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    std::vector<std::string> class_names = {
        "smoke", "phone"
    };
    yolov5->load("yolov5-lite-s-smoke-phone", 416, mean_vals, norm_vals, class_names, true);


    static Pose_Classification* pose_classification = new Pose_Classification();
    std::vector<std::string> pose_name = { "stand", "lay", "sit" };
    pose_classification->load("test-sim-opt-fp16", width, height, pose_name, true);


    static Yolo_Face* yoloface = new Yolo_Face();
    //const float mean_vals[] = { 0.0f, 0.0f, 0.0f };
    //const float norm_vals[] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    std::vector<std::string> class_namesz = {"person", "face"};
    yoloface->load("yolov8s-pose-face-person", 640, mean_vals, norm_vals, class_namesz, true);


    //static Yolo_Face* yolo_body_pose = new Yolo_Face();
    //char* class_namesz[1] = {
    //    "person"
    //};
    //yolo_body_pose->load("yolov8s-pose", 640, mean_vals, norm_vals, class_namesz, true);

    //Screenshot screenshot;
    std::chrono::system_clock::time_point start_time;


    cv::Mat frame;
    while (true)
    {
        start_time = std::chrono::system_clock::now();

        capture >> frame;
        //cv::Mat frame = screenshot.getScreenshot(0,0,1000,1000);

        cv::Mat m = frame.clone();

        //std::vector<Object_pose> objects;
        //yoloPose->detect(m, objects);
        //yoloPose->draw(frame, objects);

        std::vector<Object> objects;
        yolov5->detect(m, objects);
        yolov5->draw(frame, objects);


        //std::vector<ObjectYolov8> objects;
        //yolov8->detect(m, objects);
        //yolov8->draw(frame, objects);


        std::vector<Object_Face> objects_face;
        yoloface->detect(m, objects_face);
        pose_classification->detect(objects_face);
        yoloface->draw(frame, objects_face);

        //std::vector<Object_Face> objectsz;
        //yolo_body_pose->detect(m, objectsz);
        //yolo_body_pose->draw(frame, objectsz);

        //ncnn::Mat objects;
        //pfldPose->detect(m, objects);
        //pfldPose->draw(frame, objects);

        draw_dps(frame, start_time);

        imshow("视频", frame);
        if (cv::waitKey(1) > 0) {
            break;
        };
        //break;
    }
}