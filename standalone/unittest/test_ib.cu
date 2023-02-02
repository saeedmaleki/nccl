#include "net_ib.h"
#include <mpi.h>
#include <unistd.h>
#define bytes 1024
#define TAG 7
#define PORT 40000
#define ADDR "172.16.1.138"
int ib_send()
{
    ncclDebugLogger_t logger;
    setenv("NCCL_IB_HCA", "=mlx5_ib1:7001", 1);
    NCCLCHECK(ncclIbInit(logger));
    char *sendbuff = (char *)malloc(bytes);
    for (int i = 0; i < bytes; i++) {
        sendbuff[i] = i;
    }
    ncclIbHandle handle;
    ncclIbSendComm *sendComm;
    ncclIbListenComm *listenComm;
    handle.connectAddr.sin.sin_family = AF_INET;
    handle.connectAddr.sin.sin_port = htons(PORT);
    inet_aton(ADDR, &handle.connectAddr.sin.sin_addr);
    // ncclIbIfAddr.sin.sin_family = AF_INET;
    // ncclIbIfAddr.sin.sin_port = htons(PORT);
    // inet_aton("127.0.0.1", &ncclIbIfAddr.sin.sin_addr);
    // the sender uses ib1
    NCCLCHECK(ncclIbConnect(1, &handle, (void **)&sendComm));
    ibv_mr *mhandle;
    // NCCLCHECK(ncclIbRegMr(sendComm, sendbuff, bytes, NCCL_PTR_HOST,
    //                       (void **)&mhandle));
    // NCCLCHECK(ncclIbIsend(sendComm, sendbuff, bytes, TAG, mhandle, 0));
    printf("Send finished\n");
}

int ib_recv()
{
    ncclDebugLogger_t logger;
    setenv("NCCL_IB_HCA", "mlx5_ib2:1", 1);

    NCCLCHECK(ncclIbInit(logger));
    char *recvbuff = (char *)malloc(bytes);
    ncclIbHandle handle;
    ncclIbListenComm *listenComm;
    // handle.connectAddr.sin.sin_family = AF_INET;
    // handle.connectAddr.sin.sin_port = htons(PORT);
    // inet_aton("127.0.0.1", &handle.connectAddr.sin.sin_addr);
    // ncclIbIfAddr.sin.sin_family = AF_INET;
    // ncclIbIfAddr.sin.sin_port = htons(PORT);
    // inet_aton("127.0.0.1", &ncclIbIfAddr.sin.sin_addr);
    // printf("ncclIbIfAddr.sin.sin_addr=%s", inet_ntoa(ncclIbIfAddr.sin.sin_addr));
    // printf("ncclIbIfAddr.sin.sin_addr=%s", (ncclIbIfAddr.sin.sin_addr));
    // the receiver uses ib0
    NCCLCHECK(ncclIbListen(2, &handle, (void **)&listenComm));

    ncclIbRecvComm *recvComm;
    NCCLCHECK(ncclIbAccept(listenComm, (void **)&recvComm));
    // ibv_mr *mhandle;
    // NCCLCHECK(ncclIbRegMr(recvComm, recvbuff, bytes, NCCL_PTR_HOST,
    //                       (void **)&mhandle));
    // int size = bytes;
    // int tag = TAG;
    // NCCLCHECK(ncclIbIrecv(recvComm, 1, (void **)&recvbuff, &size, &tag,
    //                       (void **)&mhandle, 0));
    // check the recvbuff
    for (int i = 0; i < bytes; i++) {
        if (recvbuff[i] != i) {
            printf("Error: recvbuff[%d]=%d\n", i, recvbuff[i]);
            return -1;
        }
    }
    
    
    printf("Success\n");
}

int ib_sendrecv_test(int world_rank)
{
    // int devicenum;
    // ncclNetIb.devices(&devicenum);
    // printf("devicenum=%d", devicenum);
    // ncclNetProperties_t properties;
    // NCCLCHECK(ncclIbGetProperties(0, &properties));
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