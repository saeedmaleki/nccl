#include "alloc.h"
#include "net_ib.h"
#include <mpi.h>
#include <unistd.h>
#define bytes 1024
#define TAG 7
#define PORT 40000
// this ADDR is the IP address of mlx5_ib0 on my machine, set it to your own
#define ADDR "172.16.1.138"
// all of the functions of ncclNetIb is non-blocking, so we need to run them in
// a loop
int ib_send()
{
    ncclDebugLogger_t logger;
    NCCLCHECK(ncclIbInit(logger));
    char *sendbuff = NULL;
    NCCLCHECK(ncclIbMalloc((void **)&sendbuff, bytes));
    for (int i = 0; i < bytes; i++) {
        sendbuff[i] = i % 47;
    }
    ncclIbHandle handle;
    ncclIbSendComm *sendComm = NULL;
    ncclIbListenComm *listenComm;
    handle.connectAddr.sin.sin_family = AF_INET;
    handle.connectAddr.sin.sin_port = htons(PORT);
    inet_aton(ADDR, &handle.connectAddr.sin.sin_addr);
    // this magic is used to identify if the connection is established by NCCL
    handle.magic = NCCL_SOCKET_MAGIC;
    // the sender uses ib1
    while (sendComm == NULL) {
        NCCLCHECK(ncclIbConnect(1, &handle, (void **)&sendComm));
    }
    ibv_mr *mhandle;
    NCCLCHECK(ncclIbRegMr(sendComm, sendbuff, bytes, NCCL_PTR_HOST,
                          (void **)&mhandle));
    struct ncclIbRequest *requset = NULL;
    int done = 0;
    int finished_size = 0;
    while (1) {
        // the ncclIbIsend is non-blocking, so we need to run it in a loop
        NCCLCHECK(ncclIbIsend(sendComm, sendbuff, bytes, TAG, mhandle,
                              (void **)&requset));
        if (requset != 0) {
            break;
        }
    }
    while (done == 0) {
        NCCLCHECK(ncclIbTest(requset, &done, &finished_size));
    }

    printf("Send finished\n");
}

int ib_recv()
{
    ncclDebugLogger_t logger;

    NCCLCHECK(ncclIbInit(logger));
    char *recvbuff = NULL;
    NCCLCHECK(ncclIbMalloc((void **)&recvbuff, bytes));
    ncclIbHandle handle;
    ncclIbListenComm *listenComm;
    NCCLCHECK(ncclIbListen(2, &handle, (void **)&listenComm));

    ncclIbRecvComm *recvComm;
    NCCLCHECK(ncclIbAccept(listenComm, (void **)&recvComm));
    ibv_mr *mhandle;
    NCCLCHECK(ncclIbRegMr(recvComm, recvbuff, bytes, NCCL_PTR_HOST,
                          (void **)&mhandle));
    int size = bytes;
    int tag = TAG;
    struct ncclIbRequest *requset = NULL;
    NCCLCHECK(ncclIbIrecv(recvComm, 1, (void **)&recvbuff, &size, &tag,
                          (void **)&mhandle, (void **)&requset));
    int done = 0;
    int finished_size = 0;
    while (done == 0) {
        NCCLCHECK(ncclIbTest(requset, &done, &finished_size));
    }
    // check the recvbuff
    for (int i = 0; i < bytes; i++) {
        if (recvbuff[i] != i % 47) {
            printf("Error: recvbuff[%d]=%d\n", i, recvbuff[i]);
            return -1;
        }
    }
    printf("Success\n");
}

int ib_sendrecv_test(int world_rank)
{
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