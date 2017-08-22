#include <Eigen/Eigen>
#include <unsupported/Eigen/CXX11/Tensor>

#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <string>

using namespace std;
using namespace dynet;

namespace Eigen {
  struct DefaultDevice;
  class CudaStreamDevice;
  struct GpuDevice;
}

Eigen::TensorMap<Eigen::Tensor<float, 1>> tvec(float* x, int s) {
    return Eigen::TensorMap<Eigen::Tensor<float, 1>>(x, s);
}

int main2()
{
    auto dev = Eigen::DefaultDevice();
    const int size = 1000;
    float* x = new float[size];
    float* x2 = new float[size];
    tvec(x, size).device(dev) = tvec(x2, size);
    return 0;
}

vector<float> gen(int x){
    auto r = vector<float>(x);
    for(int i=0; i<x; i++)
        r[i] = i;
    return r;
}

int main(int argc, char** argv) {
    const int N = 1000;
    dynet::initialize(argc, argv);
    ParameterCollection m;
    ComputationGraph cg;
    auto s = input(cg, Dim({N,}), gen(N));
    auto ss = nobackprop(s);

    cg.forward(ss);
    return 0;
}