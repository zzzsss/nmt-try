#include "dynet/nodes.h"
#include "dynet/dynet.h"
#include "dynet/training.h"
#include "dynet/expr.h"
#include "dynet/lstm.h"
#include <iostream>

int main(int argc, char** argv) {
	dynet::initialize(argc, argv);
    ParameterCollection m;
	Parameter W= m.add_parameters({10, 10});
	ComputationGraph cg;
	Expression x = parameter(cg, W);
	auto z = W*W;
	for(int i=0; i<10000; i++){
		cg.checkpoint();
		for(int j=0; j<10000; j++){
			auto one = z+z;
		}
		cg.revert();
	}
	return 0;
}
	