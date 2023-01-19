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

__global__ void test_send(float *data_src, char *recvbuff,
                          uint64_t *sendConnHead, int size)
{
    using Proto = ProtoLL;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int sendPeers[2] = {0, -1};
    int recvPeers[2] = {0, -1};
    ncclDevChannelPeer peerInfo;
    peerInfo.send[0].buffs[NCCL_PROTO_LL] = recvbuff;
    peerInfo.send[0].head = sendConnHead;
    peerInfo.send[0].step = 0;
    // peerInfo.recv[0].buffs[NCCL_PROTO_LL] = recvbuff;
    // peerInfo.recv[0].head = sendConnHead;
    // peerInfo.recv[0].step = 0;
    Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, sendPeers, recvPeers, data_src, NULL, &peerInfo,
        ncclDevSum, 0);
    prims.send(0, size);
    return;
}

__global__ void test_recv(float *data_dst, char *recvbuff,
                          uint64_t *sendConnHead, int size)
{
    using Proto = ProtoLL;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int sendPeers[2] = {0, -1};
    int recvPeers[2] = {0, -1};
    ncclDevChannelPeer peerInfo;
    // peerInfo.send[0].buffs[NCCL_PROTO_LL] = recvbuff;
    peerInfo.send[0].head = sendConnHead;
    peerInfo.send[0].step = 0;
    peerInfo.recv[0].buffs[NCCL_PROTO_LL] = recvbuff;
    // peerInfo.recv[0].head = sendConnHead;
    peerInfo.recv[0].step = 0;
    Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, sendPeers, recvPeers, NULL, data_dst, &peerInfo,
        ncclDevSum, 0);
    prims.recv(0, size);
    return;
}

// test GPU 0 send to GPU 1
int sendrecv_test()
{
    int size = 1024;
    // There are four buffers that needs to be allocated in LL protocol. The
    // data_src is the data source buffer, located on sender GPU. The data_dst
    // is the data destination buffer, located on receiver GPU.
    float *data_src, *data_dst;
    // The recvbuff is the buffer used to receive data from sender GPU, it is
    // located on receiver GPU. The sendConnHead is the buffer used to sync the
    // sender and receiver GPU to avoid data corruption. It is located on sender
    // GPU.
    char *recvbuff;
    uint64_t *sendConnHead;
    // enable peer access
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaDeviceEnablePeerAccess(0, 0));

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMalloc(&data_src, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&sendConnHead, sizeof(uint64_t)));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMalloc(&data_dst, size * sizeof(float)));
    // currently I set recvbuff to two times of size of data to avoid error
    CUDACHECK(cudaMalloc(&recvbuff, 2 * size * sizeof(float)));
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
    test_send<<<1, 32>>>(data_src, recvbuff, sendConnHead, size);
    CUDACHECK(cudaSetDevice(1));
    test_recv<<<1, 32>>>(data_dst, recvbuff, sendConnHead, size);
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
    printf("Success\n");
    return 0;
}

int main()
{
    setbuf(stdout, NULL);
    sendrecv_test();
}