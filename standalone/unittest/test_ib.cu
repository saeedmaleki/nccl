#include "net_ib.h"
#include <mpi.h>

int ib_sendrecv_test(int world_rank)
{
    ncclDebugLogger_t logger;
    ncclIbInit(logger);
    int devicenum;
    ncclNetIb.devices(&devicenum);
    printf("devicenum=%d", devicenum);
    ncclNetProperties_t properties;
    ncclIbGetProperties(0, &properties);
    printf("name=%s, pciPath=%s, guid=%lu, ptrSupport=%d, speed=%d, port=%d, "
           "latency=%f, maxComms=%d, maxRecvs=%d\n",
           properties.name, properties.pciPath, properties.guid,
           properties.ptrSupport, properties.speed, properties.port,
           properties.latency, properties.maxComms, properties.maxRecvs);
    ncclIbHandle handle;
    ncclIbListenComm *listenComm;
    ncclIbListen(0, &handle, (void **)&listenComm);
}

int main()
{
    setbuf(stdout, NULL);
    

    MPI_Init(NULL, NULL);
    // Get the number of processes
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    // Get the rank of the process
    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    setbuf(stdout, NULL);
    ib_sendrecv_test(world_rank);
    MPI_Finalize();
}