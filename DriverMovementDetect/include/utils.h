//
// Created by 171153 on 2024/8/7.
//

#ifndef ULTIS_H
#define ULTIS_H

#include <vector>

template<
    class A,
    class B,
    template <class...> class Container,
    class... extras>
void flat_transfer(std::vector<A>& target, const Container<B, extras...>& source)
{
    for(const auto& elem : source)
    {
        if constexpr(std::is_same<A, B>::value){
            target.emplace_back(elem);
        }
        else{
            flat_transfer(target, elem);
        }
    }
}

template <
    class B,
    class A,
    template <class...> class Container,
    class... extras>
void unflat_transfer(Container<B, extras...>& target, const std::vector<A>& source, size_t& index, const std::vector<size_t>& dimensions, size_t dim = 0)
{
    target.resize(dimensions[dim]);
    for(auto& elem : target)
    {
        if constexpr(std::is_same<B, A>::value)
        {
            elem = source[index++];
        }
        else
        {
            unflat_transfer(elem, source, index, dimensions, dim + 1);
        }
    }
}

// 重载函数用于将一维向量转为多维容器
template <
    class B,
    template <class...> class Container,
    class A>
Container<B> unflat_transfer(const std::vector<A>& source, const std::vector<size_t>& dimensions)
{
    Container<B> target;
    size_t index = 0;
    unflat_transfer(target, source, index, dimensions);
    return target;
}

void softmax(vector<auto> x, int row, int column)
{
    for (int j = 0; j < row; ++j)
    {
        float max = 0.0;
        float sum = 0.0;
        for (int k = 0; k < column; ++k)
            if (max < x[k + j*column])
                max = x[k + j*column];
        for (int k = 0; k < column; ++k)
        {
            x[k + j*column] = exp(x[k + j*column] - max);    // prevent data overflow
            sum += x[k + j*column];
        }
        for (int k = 0; k < column; ++k) x[k + j*column] /= sum;
    }
}   //row*column



#endif //ULTIS_H
