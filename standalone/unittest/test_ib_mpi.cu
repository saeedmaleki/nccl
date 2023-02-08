#include "alloc.h"
#include "net_ib.h"
#include <mpi.h>
#include <unistd.h>
#define bytes 1024
#define TAG 7

// this ADDR is the IP address of mlx5_ib0 on my machine, the port is the socket
// communication port. Set it to your IP address of mlx5_ib0.
// This IP address is used for OOB(Out of Band) connection.
// NCCL chooses the first mlx5_ib0 IP address as the OOB address. In this test
// code, the receiver will listen on this IP address and port, and the sender
// will connect to this IP address and port. After the socket connection is
// established, the two sides will exchange the ibv_mr and qp to set up the IB
// connection. Then the sender will send data to the receiver using the IB.
#define PORT 40000

#define ADDR "172.16.1.138"

// all of the functions of ncclNetIb is non-blocking, so we need to run them in
// a loop
int ib_send()
{
    ncclDebugLogger_t logger;
    // gets all ib devices and stores them in ncclIbDevs
    NCCLCHECK(ncclIbInit(logger));
    char *sendbuff = NULL;
    NCCLCHECK(ncclIbMalloc((void **)&sendbuff, bytes));
    for (int i = 0; i < bytes; i++) {
        sendbuff[i] = i % 47;
    }
    ncclIbHandle handle;
    handle.connectAddr.sin.sin_family = AF_INET;
    handle.connectAddr.sin.sin_port = htons(PORT);
    inet_aton(ADDR, &handle.connectAddr.sin.sin_addr);
    // this magic is used to identify if the connection is established by NCCL
    handle.magic = NCCL_SOCKET_MAGIC;
    // the sender uses ib1
    ncclIbSendComm *sendComm = NULL;
    // call socket connect to establish the connection, exchange the qp and cq
    // and fifo
    while (sendComm == NULL) {
        // if this function returns ncclSystemError, it means that the socket
        // connect op times out, we should retry
        (ncclIbConnect(1, &handle, (void **)&sendComm));
        sleep(1);
    }
    // register the sendbuff using ibv_reg_mr
    ibv_mr *mhandle;
    // NCCL_PTR_HOST means the sendbuff is in the host memory, if it is in the
    // GPU, we need to use NCCL_PTR_CUDA
    NCCLCHECK(ncclIbRegMr(sendComm, sendbuff, bytes, NCCL_PTR_HOST,
                          (void **)&mhandle));
    struct ncclIbRequest *requset = NULL;

    while (requset == NULL) {
        // the ncclIbIsend is non-blocking, it first checks the fifo,
        // so we need to run it in a loop
        NCCLCHECK(ncclIbIsend(sendComm, sendbuff, bytes, TAG, mhandle,
                              (void **)&requset));
    }
    int done = 0;
    int finished_size = 0;
    while (done == 0) {
        // call ibv_poll_cq to check if the send requests work elements is
        // completed
        NCCLCHECK(ncclIbTest(requset, &done, &finished_size));
    }
}

int ib_recv()
{
    ncclDebugLogger_t logger;

    NCCLCHECK(ncclIbInit(logger));
    char *recvbuff = NULL;
    NCCLCHECK(ncclIbMalloc((void **)&recvbuff, bytes));
    ncclIbHandle handle;
    ncclIbListenComm *listenComm;
    // set the recv listen port
    ncclIbIfAddr.sin.sin_port = htons(PORT);
    // start socket listen
    NCCLCHECK(ncclIbListen(2, &handle, (void **)&listenComm));
    // accept the socket connection from the sender, exchange the qp and cq and
    // fifo
    ncclIbRecvComm *recvComm = NULL;
    while (recvComm == NULL) {
        NCCLCHECK(ncclIbAccept(listenComm, (void **)&recvComm));
        sleep(1);
    }
    // register the recvbuff using ibv_reg_mr
    ibv_mr *mhandle;
    NCCLCHECK(ncclIbRegMr(recvComm, recvbuff, bytes, NCCL_PTR_HOST,
                          (void **)&mhandle));
    int size = bytes;
    int tag = TAG;
    struct ncclIbRequest *requset = NULL;
    // call ibv_post_recv to post the recv request
    NCCLCHECK(ncclIbIrecv(recvComm, 1, (void **)&recvbuff, &size, &tag,
                          (void **)&mhandle, (void **)&requset));
    int done = 0;
    int finished_size = 0;
    while (done == 0) {
        // call ibv_poll_cq to check if the recv requests work elements is
        // completed
        NCCLCHECK(ncclIbTest(requset, &done, &finished_size));
    }
    // check if the recvbuff receives correct data
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