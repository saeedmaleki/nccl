#ifndef MSCCL_H_
#define MSCCL_H_

#include <stdint.h>

#define MSCCL_MAX_NUM_STEPS 256
#define MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL 32
#define MSCCL_MAX_NUM_THREAD_BLOCKS (108*2) // set this to 108 which is the number of SMs on A100
#define MSCCL_MAX_NUM_ALGOS 4

static_assert(MAXCHANNELS*MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL >= MSCCL_MAX_NUM_THREAD_BLOCKS);
static_assert(MSCCL_MAX_NUM_STEPS <= 256, "MSCCL interpreter doesn't allow for more than nthreads dependences");

#define MSCCL_INPUT_BUFFER 0
#define MSCCL_OUTPUT_BUFFER 1
#define MSCCL_SCRATCH_BUFFER 2

#define MSCCL_SEND 0
#define MSCCL_RECV 1
#define MSCCL_RECV_COPY_SEND 2
#define MSCCL_RECV_REDUCE_SEND 3
#define MSCCL_RECV_REDUCE_COPY 4
#define MSCCL_RECV_REDUCE_COPY_SEND 5
#define MSCCL_LOCAL_COPY 6
#define MSCCL_REDUCE 7
#define MSCCL_RES_ADD 8

// TODO: compress this by a lot!
struct mscclTransfer {
  int16_t srcoffset;
  int16_t dstoffset;
  uint8_t srcbuffer; // follow MSCCL_THIS_INPUT/MSCCL_THIS_OUTPUT macros
  uint8_t dstbuffer; // follow MSCCL_THIS_INPUT/MSCCL_THIS_OUTPUT macros
  int16_t depencePointer; // index to the first dependence
  int16_t numDependences; // depencePointer+numDependences indicate the last dependence
  int8_t has_dependence;
  int16_t numReductions; // number of reductions with the same dst
  int16_t reductionPointer; // where the reduction starts
  uint8_t type;
  uint8_t count;
};

struct mscclThreadBlock {
  int16_t sendpeer;
  int16_t recvpeer;
  uint16_t nsteps;
  int8_t channelId; // associated channel. -1 indicates a threadblock with only local copies
  // step is used to index into this array. transfers[step] is the addr to transfer.
  int8_t dependentBid[MSCCL_MAX_NUM_STEPS]; // -1 if not dependent on any threadblock
  int16_t dependentStep[MSCCL_MAX_NUM_STEPS];
  int16_t reductionSrcOffsets[MSCCL_MAX_NUM_STEPS]; // in case there are multiple reductions with the same dstwewqwqew
  struct mscclTransfer transfers[MSCCL_MAX_NUM_STEPS];
};

#define MSCCL_MAX_COUNT 72

struct mscclChannelInfo {
  int sendPeers[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  // nchunksForSendPeer[i][j] represents the number of times chunks are sent in counts of j-1 for threadblock i. we do not keep counts of 0.
  int nchunksForSendPeer[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL][MSCCL_MAX_COUNT];
  int nsendPeers;
  int recvPeers[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  int nchunksForRecvPeer[MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL][MSCCL_MAX_COUNT];
  int nrecvPeers;
  int nBlocksForChannel;
};

struct mscclFlag {
  uint64_t flag;
  uint64_t align[3]; // To avoid false sharing
};

// gpuId is the one that is in comm->rank
struct mscclAlgorithm {
#define MSCCL_MAX_ALGO_NAME 63
  // name of the algorithm in the XML
  char name[MSCCL_MAX_ALGO_NAME+1];
  // a flag to specify if the MSCCL algorithm is a valid one
  bool isValid;
  // the type of collective this algorithm is
  ncclFunc_t collectiveType;
  // inPlace collective
  int inPlace;
  // number of gpus in the group
  int ngpus;
  // max(#chunks in input, #chunks in output)
  int nchunksPerLoop;
  // the protocol that the algorithm needs to use
  int protocol;
  // the range of size in which this algorithm is performant
  int64_t minBytes; int64_t maxBytes;
  // bid is used as an index into this array
  struct mscclThreadBlock mscclTB[MAXCHANNELS*MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL];
  // number of channels needed by MSCCL algorithm
  int nChannels;
  // the arrays in this struct can be inferred from mscclTB. they are created to use NCCL API easily
  struct mscclChannelInfo mscclChannels[MAXCHANNELS];
  // number of scratch chunks that MSCCL will use
  int nScratchChunks;
  //Reduction Operator. If the algorithm performs reduction it will specify the reduction operator.
  //If the algorithm do not perform reduction, its reduction operator is considered as ncclSum.
  ncclRedOp_t redOp;
};

struct mscclAlgorithmShared {
  // allocate enough MSCCL flags (MSCCL_MAX_NUM_THREAD_BLOCKS_PER_CHANNEL * MAXCHANNELS) to synchronize across thread blocks
  struct mscclFlag* flags;
  // this flag is used to indicate we have we have looped around the channels work queue. Once that happens, the flags need to be reset.
  int flagsNeedReset;
  // declaration for scratchBuffer. This is only to be accessed by the host
  size_t scratchBufferSize;
  void* scratchBuffer;
};

#endif
