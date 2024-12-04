#include <tbb/flow_graph.h>
#include <iostream>
#include <tuple>

void multifunction_node()
{
    tbb::flow::graph g;

    // multifuction_node 产生不同类型的输出
    tbb::flow::multifunction_node<int, std::tuple<int, float>> node1(
        g, tbb::flow::serial, [](const int& input, tbb::flow::multifunction_node<int, std::tuple<int, float>>::output_ports_type& ports) {
            std::cout << "Node1 processing: " << input << std::endl;

            // 发送给不同的输出端口
            std::get<0>(ports).try_put(input * 10);   // 发送整数
            std::get<1>(ports).try_put(input * 1.5f); // 发送浮点数
        });

    // 两个下游节点分别接收不同的数据类型
    tbb::flow::function_node<int> node2(g, tbb::flow::serial, [](int v) {
        std::cout << "Node2 received (int): " << v << std::endl;
    });

    tbb::flow::function_node<float> node3(g, tbb::flow::serial, [](float v) {
        std::cout << "Node3 received (float): " << v << std::endl;
    });

    // 连接不同的输出端口到不同的 function_node
    tbb::flow::make_edge(tbb::flow::output_port<0>(node1), node2);
    tbb::flow::make_edge(tbb::flow::output_port<1>(node1), node3);

    // 发送任务
    node1.try_put(2);

    g.wait_for_all();
}


struct Test
{
    int a;
    int b;
};

using namespace std;
#include <thread>
int main()
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
