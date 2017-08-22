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
#include <unistd.h>
using namespace std;
using namespace dynet;
// g++ -g -std=c++11 -L ~/tmp/dynet/build/dynet/ -I ~/libs/eigen-170714/ -I ~/tmp/dynet/ test.cpp -ldynet
// LD_LIBRARY_PATH=$LD_LIBRARY_PATH:~/tmp/dynet/build/dynet ./a.out --dynet-mem 4
int main(int argc, char** argv) {
    const int ITER = 10;
    const int STEP = 50;
    const int N = 1000;
    const int BS = 4000;
	dynet::initialize(argc, argv);
    ParameterCollection m;
	Parameter W = m.add_parameters({N, N});
	vector<float> ini(N*BS, 0.5);
	for(int i=0; i<ITER; i++){
        ComputationGraph cg;
        Expression x = parameter(cg, W);
        auto start0 = input(cg, Dim({N*BS,}), ini);
        auto add0 = input(cg, Dim({N*BS,}), ini);
        auto start = reshape(start0, Dim({N,}, BS));
        auto add = reshape(add0, Dim({N,}, BS));
        for(int s=0; s<STEP; s++)
            start = x * start + add;
        auto loss = dot_product(start, add);
        auto loss2 = sum_batches(loss);
        cg.backward(loss2);
        cout << "Step " << i << endl;
        system("cat /proc/`pgrep a.out`/status | grep VmRSS");
        system("nvidia-smi 2>&1 | grep -E 'a.out.*MiB'");
	}
	return 0;
}
