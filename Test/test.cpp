#include <tbb/flow_graph.h>
#include <iostream>
#include <tuple>
#include <thread>
#include <random>
#include <queue>
void multifunction_node()
{
    tbb::flow::graph g;

    tbb::flow::source_node<int> source(g, [](int& item) -> bool {
        static int value = 0;
        if (value < 10) {
            item = value;  // {序号, 数据}
            ++value;
            return true;
        }
        return false;
    }, false);

    // multifuction_node 产生不同类型的输出
    tbb::flow::multifunction_node<int, std::tuple<int, float>> node1(
        g, tbb::flow::serial, [](const int& input, tbb::flow::multifunction_node<int, std::tuple<int, float>>::output_ports_type& ports) {
            std::cout << "Node1 processing: " << input << std::endl;
            return;
            // 发送给不同的输出端口
            std::get<0>(ports).try_put(input * 10);   // 发送整数
            if (input == 2)
            {
                std::get<1>(ports).try_put(input * 1.5f); // 发送浮点数
            }
            return;
        });

    // 两个下游节点分别接收不同的数据类型
    tbb::flow::function_node<int, int> node2(g, tbb::flow::serial, [](int v) {
        std::cout << "Node2 received (int): " << v << std::endl;
        if (v == 4)
        {
            return v;
        }
    });

    tbb::flow::function_node<int> node4(g, tbb::flow::serial, [](int v) {
    std::cout << "Node4 received (int): " << v << std::endl;
    });

    tbb::flow::function_node<float> node3(g, tbb::flow::serial, [](float v) {
        std::cout << "Node3 received (float): " << v << std::endl;
    });

    // 连接不同的输出端口到不同的 function_node
    tbb::flow::make_edge(source, node1);
    tbb::flow::make_edge(tbb::flow::output_port<0>(node1), node2);
    tbb::flow::make_edge(node2, node4);
    tbb::flow::make_edge(tbb::flow::output_port<1>(node1), node3);

    // 发送任务
    source.activate();
    g.wait_for_all();
}

struct CustomCompare {
    bool operator()(const std::pair<int,int>&a,const  std::pair<int,int>&b) const {
        return a.first > b.first; // 改为从小到大
    }
};
void priority_queue_node()
{
    tbb::flow::graph g;
    std::mt19937 gen(0);
    std::uniform_int_distribution<int> dist(1000, 10000); // 生成 1 到 100 间的整数
    // std::cout << "Radom max: " << RAND_MAX << std::endl;
    tbb::flow::source_node<std::pair<int, int>> source(g, [](std::pair<int, int>& item) -> bool {
        static int value = 0;
        if (value < 10) {
            item = {value, value};  // {序号, 数据}
            ++value;
            return true;
        }
        return false;
    }, false);

    // function_node 处理任务
    tbb::flow::function_node<std::pair<int, int>, std::pair<int, int>> process_node(
        g, tbb::flow::unlimited, [&](std::pair<int, int> input) {
            int diff = dist(gen);
            std::this_thread::sleep_for(std::chrono::milliseconds(diff));  // 模拟延迟
            std::cout << "PUT: Original=" << input.first << ", Result=" << input.second << std::endl;
            return std::make_pair(input.first, input.second * 10);        // 返回 {序号, 结果}
        });

    // priority_queue_node 确保按序输出
    tbb::flow::priority_queue_node<std::pair<int, int>> priority_node(
        g);


    // 输出处理结果
    tbb::flow::function_node<std::pair<int, int>> output_node(
        g, tbb::flow::serial, [](const std::pair<int, int>& data) {
            std::cout << "Processed: Original=" << data.first
                      << ", Result=" << data.second << std::endl;
        });

    // 建立边
    tbb::flow::make_edge(source, process_node);
    tbb::flow::make_edge(process_node, priority_node);
    tbb::flow::make_edge(priority_node, output_node);

    // 激活图
    source.activate();
    g.wait_for_all();
}
#include <atomic>
void wait_source()
{
    tbb::flow::graph g;

    std::atomic<bool> condition_met(false);

    tbb::flow::source_node<int> source(g, [&](int& output) -> bool {
        if (!condition_met.load()) {
            return false; // 暂时不生成任务
        }

        static int value = 0;
        if (value < 10) {
            output = value++;
            return true;
        }
        return false; // 停止生成任务
    }, false);

    tbb::flow::function_node<int> func(g, tbb::flow::unlimited, [](int input) {
        std::cout << "Processing: " << input << std::endl;
    });

    tbb::flow::make_edge(source, func);

    source.activate();

    std::thread condition_thread([&]() {
        std::this_thread::sleep_for(std::chrono::seconds(2));
        condition_met = true; // 改变条件
    });

    g.wait_for_all();
    condition_thread.join();

}


struct Test
{
    int a;
    int b;
};

using namespace std;

void Test()
{
    vector<float> input_tensor{1,2,3,4,5};
    vector<float> output_tensor;
    output_tensor.resize(10);

    std::thread t1([&]
    {
        memcpy(output_tensor.data(), input_tensor.data(), 5 * sizeof(float));
    });

    std::thread t2([&]
    {
        memcpy(output_tensor.data() + 5, input_tensor.data(), 5 * sizeof(float));
    });
    cout << "waht" << endl;
    t1.join();
    t2.join();

    for (auto i : output_tensor)
    {
        cout << i << "   ";
    }

}

#include <thread>
int main()
{
    wait_source();
}
