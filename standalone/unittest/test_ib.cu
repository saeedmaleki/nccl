#include "net_ib.h"
#include <mpi.h>
#define bytes 1024
int ib_send()
{
    char *sendbuff = (char *)malloc(bytes);
    for (int i = 0; i < bytes; i++) {
        sendbuff[i] = i;
    }
    ncclIbHandle handle;
    ncclIbSendComm *sendComm;
    NCCLCHECK(ncclIbConnect(1, &handle, (void **)&sendComm));
    ibv_mr *mhandle;
    NCCLCHECK(ncclIbRegMr(sendComm, sendbuff, bytes, NCCL_PTR_HOST,
                          (void **)&mhandle));
    // NCCLCHECK(ncclIbIsend(sendComm, sendbuff, bytes, 0, 0, 0));
}

int ib_recv()
{
    char *recvbuff = (char *)malloc(bytes);
    ncclIbHandle handle;
    ncclIbListenComm *listenComm;

    NCCLCHECK(ncclIbListen(0, &handle, (void **)&listenComm));
    ncclIbRecvComm *recvComm;
    NCCLCHECK(ncclIbAccept(listenComm, (void **)&recvComm));
    ibv_mr *mhandle;
    NCCLCHECK(ncclIbRegMr(recvComm, recvbuff, bytes, NCCL_PTR_HOST,
                          (void **)&mhandle));
    // NCCLCHECK(ncclIbIrecv(recvComm, recvbuff, bytes, 0, 0, 0));
}

int ib_sendrecv_test(int world_rank)
{
    printf("ib_sendrecv_test rankd:%d\n", world_rank);
    ncclDebugLogger_t logger;
    NCCLCHECK(ncclIbInit(logger));
    int devicenum;
    ncclNetIb.devices(&devicenum);
    printf("devicenum=%d", devicenum);
    ncclNetProperties_t properties;
    NCCLCHECK(ncclIbGetProperties(0, &properties));
    if (world_rank == 0) {
        ib_send();
    } else if (world_rank == 1) {
        ib_recv();
    }
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