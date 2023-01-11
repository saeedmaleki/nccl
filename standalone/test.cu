#include "devcomm.h"
#include "primitives.h"

__global__ void test(float* input, float* output, int size) {
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS/ALLREDUCE_SLICESTEPS, ALLREDUCE_SLICESTEPS>;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims
        (tid, nthreads, NULL, NULL, NULL, NULL, ncclDevSum);
    return;
}

int main(){
    test<<<32,32>>>(NULL,NULL, 0);
}