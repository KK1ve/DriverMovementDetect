
#include "DriverActionDetect.h"


void draw_dps(cv::Mat frame, std::chrono::system_clock::time_point start_time) {
    char fps[100];
    std::chrono::system_clock::time_point end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = end_time - start_time;
    sprintf(fps, "FPS: %.1f", (float)1 / (diff.count()));
    cv::putText(frame, fps, cv::Point(5, 30), cv::FONT_HERSHEY_SIMPLEX, 0.75, { 255, 255, 255 }, 1);
}


std::vector<std::variant<float, int>> process_img(ncnn::Mat& in_pad, cv::Mat rgb, int target_size, const float *norm_vals) {
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
    ncnn::copy_make_border(in,
        in_pad,
        top,
        bottom,
        left,
        right,
        ncnn::BORDER_CONSTANT,
        114.f);

    in_pad.substract_mean_normalize(0, norm_vals);

    std::vector<std::variant<float, int>> result = { height , width , (float)hpad / 2 , (float)wpad / 2 , scale };
    return result;
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
    //std::vector<std::string> class_names = {
    //    "applauding", "playing_guitar", "holding_an_umbrella", "pouring_liquid", "riding_a_bike", "pushing_a_cart", 
    //    "feeding_a_horse", "texting_message", "fishing", "playing_violin", "throwing_frisby", "cutting_vegetables", 
    //    "riding_a_horse", "taking_photos", "watching_TV", "shooting_an_arrow", "reading", "walking_the_dog", 
    //    "looking_through_a_telescope", "drinking", "brushing_teeth", "jumping", "cleaning_the_floor", "climbing", 
    //    "using_a_computer", "rowing_a_boat", "fixing_a_car", "gardening", "writing_on_a_board", "blowing_bubbles", 
    //    "cutting_trees", "washing_dishes", "waving_hands", "running", "phoning", "cooking", "looking_through_a_microscope", 
    //    "writing_on_a_book", "smoking", "fixing_a_bike"
    //};
    std::vector<std::string> class_names = {
        "drinking", "texting_message", "waving_hands", "phoning", "smoking", "normal"
    };
    yolov8->load("model.ncnn", target_size, mean_vals, norm_vals, class_names, true);


    static Pose_Classification* pose_classification = new Pose_Classification();
    std::vector<std::string> pose_name = { "up", "bending", "fall" };
    pose_classification->load("test-sim-opt-fp16", width, height, pose_name, true);


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

    std::vector<std::variant<float, int>> _result;


    std::vector<std::thread> threads;
    ncnn::Mat _in_pad;

    threads.push_back(std::thread([&] {
        capture >> frame;
        cv::Mat m = frame.clone();
        _result.clear();
        _result = process_img(_in_pad, m, target_size, norm_vals);
        }));

    while (true)
    {
        threads[0].join();
        threads.clear();

        std::vector<std::variant<float, int>> result = _result;

        ncnn::Mat in_pad = _in_pad.clone();
        start_time = std::chrono::system_clock::now();
        threads.push_back(std::thread([&] {
            capture >> frame;
            cv::Mat m = frame.clone();
            _result.clear();
            _result = process_img(_in_pad, m, target_size, norm_vals);
            }));
        //cv::Mat frame = screenshot.getScreenshot(0,0,1000,1000);


        //std::vector<Object_pose> objects;
        //yoloPose->detect(m, objects);
        //yoloPose->draw(frame, objects);

        std::vector<ObjectYolov8> objects;
        yolov8->detect(in_pad, objects, result);
        yolov8->draw(frame, objects);


        std::vector<Object_Pose> objects_face;
        yolopose->detect(in_pad, objects_face, result);
        pose_classification->detect(objects_face);
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