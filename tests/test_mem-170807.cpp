#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>

using namespace std;
using namespace dynet;

// how to compile
// g++ -g -std=c++11 -L ~/tmp/zzsdynet/build/dynet/ -I ~/libs/eigen-170714/ -I ~/tmp/zzsdynet/ test_mem-170807.cpp -ldynet

int print_mem(){
    ifstream fin;
    fin.open("/proc/self/statm");
    int x;
    fin >> x; fin >> x;
    fin.close();
    cout << "CPU:" << x*4/1024 << endl;
    system("nvidia-smi 2>&1 | grep -E 'a.out.*MiB'");
}

int main(int argc, char** argv) {
    const int ITER = 100;
    const int STEP = 10000;
    const int N = 1000;
	dynet::initialize(argc, argv);
    ParameterCollection m;
	Parameter W = m.add_parameters({N, N});
	for(int i=0; i<ITER; i++){
        ComputationGraph cg;
        Expression x = parameter(cg, W);
        auto start = input(cg, Dim({N,}, i), vector<float>(N*i));
        for(int s=0; s<STEP; s++){
            start = x * start;
            start.value();
        }
        auto loss = dot_product(start, start);
        auto loss2 = sum_batches(loss);
        cg.backward(loss2);
        // print memory stat
        print_mem();
	}
	return 0;
}
