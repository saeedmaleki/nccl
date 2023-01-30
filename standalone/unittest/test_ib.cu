#include "devcomm.h"

#define CUDACHECK(cmd)                                                         \
    do {                                                                       \
        cudaError_t e = cmd;                                                   \
        if (e != cudaSuccess) {                                                \
            printf("Failed: Cuda error %s:%d '%s' ", __FILE__, __LINE__,       \
                   cudaGetErrorString(e));                                     \
            exit(EXIT_FAILURE);                                                \
        }                                                                      \
    } while (0)

__global__ void test_send_simple(float *data_src, float *data_dst, char *buff,
                                 uint64_t *head, uint64_t *tail, int size)
{
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS,
                              ALLREDUCE_SLICESTEPS>;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int sendPeers[2] = {0, -1};
    int recvPeers[2] = {-1, -1};
    ncclDevChannelPeer peerInfo;
    peerInfo.send[0].buffs[NCCL_PROTO_SIMPLE] = buff;
    peerInfo.send[0].head = head;
    peerInfo.send[0].tail = tail;
    peerInfo.send[0].step = 0;
    peerInfo.send[0].sizesFifo = NULL;
    peerInfo.send[0].offsFifo = NULL;
    Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, recvPeers, sendPeers, data_src, NULL, &peerInfo,
        ncclDevSum, 0);
    prims.send(0, size);
    return;
}

__global__ void test_recv_simple(float *data_src, float *data_dst, char *buff,
                                 uint64_t *head, uint64_t *tail, int size)
{
    using Proto = ProtoSimple<ALLREDUCE_CHUNKSTEPS / ALLREDUCE_SLICESTEPS,
                              ALLREDUCE_SLICESTEPS>;
    int tid = threadIdx.x;
    int nthreads = blockDim.x;
    int sendPeers[2] = {-1, -1};
    int recvPeers[2] = {0, -1};
    ncclDevChannelPeer peerInfo;
    peerInfo.recv[0].buffs[NCCL_PROTO_SIMPLE] = buff;
    peerInfo.recv[0].head = head;
    peerInfo.recv[0].tail = tail;
    peerInfo.recv[0].step = 0;
    peerInfo.recv[0].sizesFifo = NULL;
    peerInfo.recv[0].offsFifo = NULL;
    Primitives<float, FuncSum<float>, FanSymmetric<1>, 1, Proto, 0> prims(
        tid, nthreads, recvPeers, sendPeers, NULL, data_dst, &peerInfo,
        ncclDevSum, 0);
    prims.recv(0, size);
    return;
}

// test GPU 0 send to GPU 1 using simple protocol
int sendrecv_test_simple()
{
    int size = 1024;
    // There are five buffers that needs to be allocated in Simple protocol. The
    // data_src is the data source buffer, located on sender GPU. The data_dst
    // is the data destination buffer, located on receiver GPU.
    float *data_src, *data_dst;
    char *buffs;    // Local for recv, remote for send
    uint64_t *tail; // Local for recv, remote for send
    uint64_t *head; // Local for send, remote for recv
    // enable peer access
    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaDeviceEnablePeerAccess(1, 0));
    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaDeviceEnablePeerAccess(0, 0));

    CUDACHECK(cudaSetDevice(0));
    CUDACHECK(cudaMalloc(&data_src, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&head, sizeof(uint64_t)));

    CUDACHECK(cudaSetDevice(1));
    CUDACHECK(cudaMalloc(&data_dst, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&buffs, size * sizeof(float)));
    CUDACHECK(cudaMalloc(&tail, sizeof(uint64_t)));
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
    test_send_simple<<<1, 32>>>(data_src, data_dst, buffs, head, tail, size);
    CUDACHECK(cudaSetDevice(1));
    test_recv_simple<<<1, 32>>>(data_src, data_dst, buffs, head, tail, size);
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
    sendrecv_test_simple();
}
