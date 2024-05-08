//
// Created by wangke on 2023/5/15.
//

#include "pose-classification.h"
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <cpu.h>


Pose_Classification::Pose_Classification()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}

int Pose_Classification::load(const char* modeltype, const int _img_width, const int _img_height, std::vector<std::string> _pose_name, bool use_gpu) {

    pose_classification.clear();
    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    pose_classification.opt = ncnn::Option();

#if NCNN_VULKAN
    pose_classification.opt.use_vulkan_compute = use_gpu;
#endif

    pose_classification.opt.num_threads = ncnn::get_big_cpu_count();
    pose_classification.opt.blob_allocator = &blob_pool_allocator;
    pose_classification.opt.workspace_allocator = &workspace_pool_allocator;

    char parampath[256];
    char modelpath[256];
    sprintf(parampath, "%s/model_zoo/%s.param", CURRENT_DIR, modeltype);
    sprintf(modelpath, "%s/model_zoo/%s.bin", CURRENT_DIR, modeltype);

    pose_classification.load_param(parampath);
    pose_classification.load_model(modelpath);

    img_height = _img_height;
    img_width = _img_width;
    pose_name = _pose_name;

    return 0;
}



int Pose_Classification::detect(std::vector<Object_Face>& objects) {
    for (int z = 0; z < objects.size(); z++) {
        Object_Face of = objects[z];
        if (of.label != 0) continue;
        const int num_points = 17;
        ncnn::Mat in(1, num_points * 3);
        for (int i = 0; i < num_points; i++) {
            in[i] = of.kps[i] / img_width;
            in[i+1] = of.kps[i+1] / img_height;
            in[i+2] = of.kps[i+1];
        }

        ncnn::Extractor ex = pose_classification.create_extractor();

        ex.input("input", in);
        ncnn::Mat out;
        ex.extract("output", out);

        float max = 0;
        int max_index = 0;

        std::vector<float> table(3,0);


        for (int q = 0; q < out.c; q++)
        {
            const float* ptr = out.channel(q);

            for (int y = 0; y < out.h; y++)
            {
                for (int x = 0; x < out.w; x++)
                {
                    table[x] = ptr[x];
                }
                ptr += out.w;

            }

        }


        
        for (int i = 0; i < table.size(); i++) {
            if (table[i] > max) {
                max = table[i];
                max_index = i;
            }
        }
        objects[z].pose_name = pose_name[max_index];

    }
   
    return 0;
}


