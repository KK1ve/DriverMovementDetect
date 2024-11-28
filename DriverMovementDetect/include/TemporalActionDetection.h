//
// Created by 171153 on 2024/8/6.
//

#ifndef TEMPORALACTIONDETECTION_H
#define TEMPORALACTIONDETECTION_H

#include <fstream>
#include <map>
#include <utils.h>
#include <opencv2/highgui.hpp>
#include <vector>
#include "bmnn_utils.h"
using namespace cv;
using namespace std;



class TAD
{
    public:
        TAD(const string& modelpath, int _origin_h = 1920, int _origin_w = 1080, float nms_thresh_ = 0.5, float conf_thresh_ = 0.1, int rate = 3);
        std::map<unsigned long, vector<float>> detect_multi_hot(const std::map<size_t, Mat>& video_mat_with_track_id);
        void detect_one_hot(const Mat& video_mat, vector<ObjectPose> &boxes);
        Mat vis_one_hot(Mat frame, vector<Bbox> boxes, vector<float> det_conf, vector<vector<float>> cls_conf, vector<int> keep_inds, float vis_thresh);
        Mat vis(const Mat& frame, const vector<ObjectPose>& boxes, bool show_action = true, float action_thresh = 0.03, float keypoint_thresh = 0.4f);
        void clear_clips_cache();

        int len_clip;
        vector<float> diffs;

    private:
        int origin_h;
        int origin_w;
        std::chrono::system_clock::time_point start_time;
        std::chrono::system_clock::time_point end_time;
        std::chrono::duration<float> diff;
        int batch_size = 1;
        vector<float> input_tensor;
        void preprocess(vector<Mat> video_clip);
        Mat preprocess(const Mat& video_mat);
        int inpBatchSize;
        int inpWidth;
        int inpHeight;
        float nms_thresh;
        float conf_thresh;

        int rate;
        vector<vector<Mat>> video_clips;
        std::map<size_t, vector<vector<Mat>>> multi_video_clips;
        int current_rate = 0;
        const int topk = 40;
        const int strides[3] = {8, 16, 32};
        vector<float> means{0.485, 0.456, 0.406};
        vector<float> stds{0.229, 0.224, 0.225};
        void generate_proposal_one_hot(int stride, const float *conf_pred, const float *cls_pred, const float *reg_pred, vector<ObjectPose> &boxes);
        vector<char *> input_names;
        vector<char *> output_names;
        vector<vector<int64_t>> input_node_dims;  // >=1 outputs
        vector<vector<int64_t>> output_node_dims; // >=1 outputs
        std::shared_ptr<BMNNContext> m_bmContext;
        std::shared_ptr<BMNNNetwork> m_bmNetwork;
        std::shared_ptr<BMNNTensor>  m_input_tensor;
        std::shared_ptr<BMNNTensor>  m_output_tensor;
        bm_tensor_t bm_input_tensor;

        const char *labels[80] = {"bend/bow(at the waist)", "crawl", "crouch/kneel", "dance", "fall down", "get up", "jump/leap", "lie/sleep", "martial art", "run/jog", "sit", "stand", "swim", "walk", "answer phone", "brush teeth", "carry/hold (an object)", "catch (an object)", "chop", "climb (e.g. a mountain)", "clink glass", "close (e.g., a door, a box)", "cook", "cut", "dig", "dress/put on clothing", "drink", "drive (e.g., a car, a truck)", "eat", "enter", "exit", "extract", "fishing", "hit (an object)", "kick (an object)", "lift/pick up", "listen (e.g., to music)", "open (e.g., a window, a car door)", "paint", "play board game", "play musical instrument", "play with pets", "point to (an object)", "press", "pull (an object)", "push (an object)", "put down", "read", "ride (e.g., a bike, a car, a horse)", "row boat", "sail boat", "shoot", "shovel", "smoke", "stir", "take a photo", "text on/look at a cellphone", "throw", "touch (an object)", "turn (e.g., a screwdriver)", "watch (e.g., TV)", "work on a computer", "write", "fight/hit (a person)", "give/serve (an object) to (a person)", "grab (a person)", "hand clap", "hand shake", "hand wave", "hug (a person)", "kick (a person)", "kiss (a person)", "lift (a person)", "listen to (a person)", "play with kids", "push (another person)", "sing to (e.g., self, a person, a group)", "take (an object) from (a person)", "talk to (e.g., self, a person, a group)", "watch (a person)"};
        const int num_class = 80;

        /*
        const char *ucf24_labels[] = {"Basketball", "BasketballDunk", "Biking", "CliffDiving", "CricketBowling", "Diving", "Fencing", "FloorGymnastics", "GolfSwing", "HorseRiding", "IceDancing", "LongJump", "PoleVault", "RopeClimbing", "SalsaSpin", "SkateBoarding", "Skiing", "Skijet", "SoccerJuggling", "Surfing", "TennisSwing", "TrampolineJumping", "VolleyballSpiking", "WalkingWithDog"};
        const char *ava_labels[80] = {"bend/bow(at the waist)", "crawl", "crouch/kneel", "dance", "fall down", "get up", "jump/leap", "lie/sleep", "martial art", "run/jog", "sit", "stand", "swim", "walk", "answer phone", "brush teeth", "carry/hold (an object)", "catch (an object)", "chop", "climb (e.g. a mountain)", "clink glass", "close (e.g., a door, a box)", "cook", "cut", "dig", "dress/put on clothing", "drink", "drive (e.g., a car, a truck)", "eat", "enter", "exit", "extract", "fishing", "hit (an object)", "kick (an object)", "lift/pick up", "listen (e.g., to music)", "open (e.g., a window, a car door)", "paint", "play board game", "play musical instrument", "play with pets", "point to (an object)", "press", "pull (an object)", "push (an object)", "put down", "read", "ride (e.g., a bike, a car, a horse)", "row boat", "sail boat", "shoot", "shovel", "smoke", "stir", "take a photo", "text on/look at a cellphone", "throw", "touch (an object)", "turn (e.g., a screwdriver)", "watch (e.g., TV)", "work on a computer", "write", "fight/hit (a person)", "give/serve (an object) to (a person)", "grab (a person)", "hand clap", "hand shake", "hand wave", "hug (a person)", "kick (a person)", "kiss (a person)", "lift (a person)", "listen to (a person)", "play with kids", "push (another person)", "sing to (e.g., self, a person, a group)", "take (an object) from (a person)", "talk to (e.g., self, a person, a group)", "watch (a person)"};
        const char *kinetics-400[400] = {"abseiling", "air drumming", "answering questions", "applauding", "applying cream", "archery", "arm wrestling", "arranging flowers", "assembling computer", "auctioning", "baby waking up", "baking cookies", "balloon blowing", "bandaging", "barbequing", "bartending", "beatboxing", "bee keeping", "belly dancing", "bench pressing", "bending back", "bending metal", "biking through snow", "blasting sand", "blowing glass", "blowing leaves", "blowing nose", "blowing out candles", "bobsledding", "bookbinding", "bouncing on trampoline", "bowling", "braiding hair", "breading or breadcrumbing", "breakdancing", "brush painting", "brushing hair", "brushing teeth", "building cabinet", "building shed", "bungee jumping", "busking", "canoeing or kayaking", "capoeira", "carrying baby", "cartwheeling", "carving pumpkin", "catching fish", "catching or throwing baseball", "catching or throwing frisbee", "catching or throwing softball", "celebrating", "changing oil", "changing wheel", "checking tires", "cheerleading", "chopping wood", "clapping", "clay pottery making", "clean and jerk", "cleaning floor", "cleaning gutters", "cleaning pool", "cleaning shoes", "cleaning toilet", "cleaning windows", "climbing a rope", "climbing ladder", "climbing tree", "contact juggling", "cooking chicken", "cooking egg", "cooking on campfire", "cooking sausages", "counting money", "country line dancing", "cracking neck", "crawling baby", "crossing river", "crying", "curling hair", "cutting nails", "cutting pineapple", "cutting watermelon", "dancing ballet", "dancing charleston", "dancing gangnam style", "dancing macarena", "deadlifting", "decorating the christmas tree", "digging", "dining", "disc golfing", "diving cliff", "dodgeball", "doing aerobics", "doing laundry", "doing nails", "drawing", "dribbling basketball", "drinking", "drinking beer", "drinking shots", "driving car", "driving tractor", "drop kicking", "drumming fingers", "dunking basketball", "dying hair", "eating burger", "eating cake", "eating carrots", "eating chips", "eating doughnuts", "eating hotdog", "eating ice cream", "eating spaghetti", "eating watermelon", "egg hunting", "exercising arm", "exercising with an exercise ball", "extinguishing fire", "faceplanting", "feeding birds", "feeding fish", "feeding goats", "filling eyebrows", "finger snapping", "fixing hair", "flipping pancake", "flying kite", "folding clothes", "folding napkins", "folding paper", "front raises", "frying vegetables", "garbage collecting", "gargling", "getting a haircut", "getting a tattoo", "giving or receiving award", "golf chipping", "golf driving", "golf putting", "grinding meat", "grooming dog", "grooming horse", "gymnastics tumbling", "hammer throw", "headbanging", "headbutting", "high jump", "high kick", "hitting baseball", "hockey stop", "holding snake", "hopscotch", "hoverboarding", "hugging", "hula hooping", "hurdling", "hurling (sport)", "ice climbing", "ice fishing", "ice skating", "ironing", "javelin throw", "jetskiing", "jogging", "juggling balls", "juggling fire", "juggling soccer ball", "jumping into pool", "jumpstyle dancing", "kicking field goal", "kicking soccer ball", "kissing", "kitesurfing", "knitting", "krumping", "laughing", "laying bricks", "long jump", "lunge", "making a cake", "making a sandwich", "making bed", "making jewelry", "making pizza", "making snowman", "making sushi", "making tea", "marching", "massaging back", "massaging feet", "massaging legs", "massaging person's head", "milking cow", "mopping floor", "motorcycling", "moving furniture", "mowing lawn", "news anchoring", "opening bottle", "opening present", "paragliding", "parasailing", "parkour", "passing American football (in game)", "passing American football (not in game)", "peeling apples", "peeling potatoes", "petting animal (not cat)", "petting cat", "picking fruit", "planting trees", "plastering", "playing accordion", "playing badminton", "playing bagpipes", "playing basketball", "playing bass guitar", "playing cards", "playing cello", "playing chess", "playing clarinet", "playing controller", "playing cricket", "playing cymbals", "playing didgeridoo", "playing drums", "playing flute", "playing guitar", "playing harmonica", "playing harp", "playing ice hockey", "playing keyboard", "playing kickball", "playing monopoly", "playing organ", "playing paintball", "playing piano", "playing poker", "playing recorder", "playing saxophone", "playing squash or racquetball", "playing tennis", "playing trombone", "playing trumpet", "playing ukulele", "playing violin", "playing volleyball", "playing xylophone", "pole vault", "presenting weather forecast", "pull ups", "pumping fist", "pumping gas", "punching bag", "punching person (boxing)", "push up", "pushing car", "pushing cart", "pushing wheelchair", "reading book", "reading newspaper", "recording music", "riding a bike", "riding camel", "riding elephant", "riding mechanical bull", "riding mountain bike", "riding mule", "riding or walking with horse", "riding scooter", "riding unicycle", "ripping paper", "robot dancing", "rock climbing", "rock scissors paper", "roller skating", "running on treadmill", "sailing", "salsa dancing", "sanding floor", "scrambling eggs", "scuba diving", "setting table", "shaking hands", "shaking head", "sharpening knives", "sharpening pencil", "shaving head", "shaving legs", "shearing sheep", "shining shoes", "shooting basketball", "shooting goal (soccer)", "shot put", "shoveling snow", "shredding paper", "shuffling cards", "side kick", "sign language interpreting", "singing", "situp", "skateboarding", "ski jumping", "skiing (not slalom or crosscountry)", "skiing crosscountry", "skiing slalom", "skipping rope", "skydiving", "slacklining", "slapping", "sled dog racing", "smoking", "smoking hookah", "snatch weight lifting", "sneezing", "sniffing", "snorkeling", "snowboarding", "snowkiting", "snowmobiling", "somersaulting", "spinning poi", "spray painting", "spraying", "springboard diving", "squat", "sticking tongue out", "stomping grapes", "stretching arm", "stretching leg", "strumming guitar", "surfing crowd", "surfing water", "sweeping floor", "swimming backstroke", "swimming breast stroke", "swimming butterfly stroke", "swing dancing", "swinging legs", "swinging on something", "sword fighting", "tai chi", "taking a shower", "tango dancing", "tap dancing", "tapping guitar", "tapping pen", "tasting beer", "tasting food", "testifying", "texting", "throwing axe", "throwing ball", "throwing discus", "tickling", "tobogganing", "tossing coin", "tossing salad", "training dog", "trapezing", "trimming or shaving beard", "trimming trees", "triple jump", "tying bow tie", "tying knot (not on a tie)", "tying tie", "unboxing", "unloading truck", "using computer", "using remote controller (not gaming)", "using segway", "vault", "waiting in line", "walking the dog", "washing dishes", "washing feet", "washing hair", "washing hands", "water skiing", "water sliding", "watering plants", "waxing back", "waxing chest", "waxing eyebrows", "waxing legs", "weaving basket", "welding", "whistling", "windsurfing", "wrapping present", "wrestling", "writing", "yawning", "yoga", "zumba"}
        */

};


/*
std::vector<int> multiclass_nms_class_aware(const std::vector<Bbox> boxes, const std::vector<float> confidences, const std::vector<int> labels, const float nms_thresh, const int num_classes)
{
    const int num_box = boxes.size();
    std::vector<int> keep(num_box, 0);
    for (int i = 0; i < num_classes; i++)
    {
        std::vector<int> inds;
        std::vector<Bbox> c_bboxes;
        std::vector<float> c_scores;
        for (int j = 0; j < labels.size(); j++)
        {
            if (labels[j] == i)
            {
                inds.emplace_back(j);
                c_bboxes.emplace_back(boxes[j]);
                c_scores.emplace_back(confidences[j]);
            }
        }
        if (inds.size() == 0)
        {
            continue;
        }

        std::vector<int> c_keep = multiclass_nms_class_agnostic(c_bboxes, c_scores, nms_thresh);
        for (int j = 0; j < c_keep.size(); j++)
        {
            keep[inds[c_keep[j]]] = 1;
        }
    }

    std::vector<int> keep_inds;
    for (int i = 0; i < keep.size(); i++)
    {
        if (keep[i] > 0)
        {
            keep_inds.emplace_back(i);
        }
    }
    return keep_inds;
}
*/












#endif //TEMPORALACTIONDETECTION_H
