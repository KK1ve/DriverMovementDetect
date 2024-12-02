#include <tbb/flow_graph.h>
#include <iostream>
#include <vector>
#include <algorithm>

int main() {
    tbb::flow::graph g;

    tbb::flow::buffer_node<int> buffer(g);

    // 插入数据
    buffer.try_put(10);
    buffer.try_put(25);
    buffer.try_put(15);
    buffer.try_put(35);

    // 按需取出元素
    tbb::flow::function_node<void> filter_node(
        g, tbb::flow::serial, [&]() {
            std::vector<int> elements;
            int item;

            // 获取所有数据
            while (buffer.try_get(item)) {
                elements.push_back(item);
            }

            // 筛选大于20的元素
            for (auto& elem : elements) {
                if (elem > 20) {
                    std::cout << "Filtered element: " << elem << std::endl;
                } else {
                    // 将未满足条件的数据重新放回缓冲区
                    buffer.try_put(elem);
                }
            }
        });

    g.wait_for_all();

    return 0;
}
