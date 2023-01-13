#include "devcomm.h"
#include "primitives.h"

#define CUDACHECK(cmd)                                                         \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("Failed: Cuda error %s:%d '%s' ", __FILE__, __LINE__,       \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void test(float *input, float *output, int size)
{
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS,
                              ALLREDUCE_SLICESTEPS>;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, NULL, NULL, NULL, NULL, ncclDevSum);
    return;
}

__global__ void test_send(float *data_src, int size)
{
    using Proto = ProtoLL;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int sendPeers[2] = {1, -1};
    int recvPeers[2] = {0, -1};
    Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, sendPeers, recvPeers, data_src, NULL, ncclDevSum);
    return;
}

__global__ void test_recv(float *data_dst, int size)
{
    using Proto = ProtoLL;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int sendPeers[2] = {0, -1};
    int recvPeers[2] = {1, -1};
    Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, sendPeers, recvPeers, NULL, data_dst, ncclDevSum);
    return;
}

int sendrecv_test()
{
    int size = 1024;
    float *data_src, *data_dst;
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMalloc(&data_src, size * sizeof(float)));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMalloc(&data_dst, size * sizeof(float)));
    float *h_data_src = (float *)malloc(size * sizeof(float));
    float *h_data_dst = (float *)malloc(size * sizeof(float));
    for (int i = 0; i < size; i++) {
        h_data_src[i] = rand() % 100;
        h_data_dst[i] = 0;
    }
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMemcpy(data_src, h_data_src, size * sizeof(float),
                         cudaMemcpyHostToDevice));
    CUDACHECK(cudaSetDevice(0));
    test_send<<<1, 1>>>(data_src, size);
    CUDACHECK(cudaSetDevice(1));
    test_recv<<<1, 1>>>(data_dst, size);
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaDeviceSynchronize());
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMemcpy(h_data_dst, data_dst, size * sizeof(float),
                         cudaMemcpyDeviceToHost));
    for (int i = 0; i < size; i++) {
        if (h_data_dst[i] != h_data_src[i]) {
            printf("Error: h_data_dst[%d] = %f != %f ", i, h_data_dst[i],
                   h_data_src[i]);
            return -1;
        }
    }
    return 0;
}

int main()
{
    setbuf(stdout, NULL);
    sendrecv_test();
}