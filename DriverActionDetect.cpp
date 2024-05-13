
#include "DriverActionDetect.h"


void draw_dps(cv::Mat frame, std::chrono::system_clock::time_point start_time) {
    char fps[100];
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    sprintf(fps, "FPS: %.1f", (float)1 / (diff.count()));
    cv::putText(frame, fps, cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.75, { 255, 255, 255 }, 1);
}


ncnn::Mat process_img(cv::Mat rgb, int target_size, const float *norm_vals) {
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_BGR2RGB, width, height, w, h);

    // pad to target_size rectangle
    int wpad = std::abs(w - target_size);
    int hpad = std::abs(h - target_size);

    int top = hpad / 2;
    int bottom = hpad - hpad / 2;
    int left = wpad / 2;
    int right = wpad - wpad / 2;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in,
        in_pad,
        top,
        bottom,
        left,
        right,
        ncnn::BORDER_CONSTANT,
        114.f);

    in_pad.substract_mean_normalize(0, norm_vals);
    return in_pad;
}

int main(int argc, char** argv)
{
    int height = 640;
    int width = 480;
    cv::VideoCapture capture;
    capture.open(0);  //修改这个参数可以选择打开想要用的摄像头

    const float mean_vals[] = { 0.0f, 0.0f, 0.0f };
    const float norm_vals[] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };

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

    const int target_size = 640;


    static Yolov8* yolov8 = new Yolov8();
    std::vector<std::string> class_names = {
        "sitting", "using_laptop","hugging","sleeping","drinking","clapping","dancing",
        "cycling","calling","laughing","eating","fighting","listening_to_music",
        "running","texting"
    };
    yolov8->load("model.ncnn", target_size, mean_vals, norm_vals, class_names, true);


    //static Pose_Classification* pose_classification = new Pose_Classification();
    //std::vector<std::string> pose_name = { "stand", "lay", "sit" };
    //pose_classification->load("test-sim-opt-fp16", width, height, pose_name, true);


    static Yolo_Pose* yolopose = new Yolo_Pose();
    std::vector<std::string> class_namesz = {"person", "face"};
    yolopose->load("yolov8s-pose-face-person", target_size, mean_vals, norm_vals, class_namesz, true);


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

        //std::vector<ObjectYolov8> objects;
        //yolov8->detect(m, objects);
        //yolov8->draw(frame, objects);



        std::vector<Object_Pose> objects_face;
        yolopose->detect(m, objects_face);
        //pose_classification->detect(objects_face);
        yolopose->draw(frame, objects_face);

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