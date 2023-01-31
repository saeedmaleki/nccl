#include "net_ib.h"

int main()
{
    setbuf(stdout, NULL);
    ncclDebugLogger_t logger;
    ncclNetIb.init(logger);
    int devicenum;
    ncclNetIb.devices(&devicenum);
    printf("devicenum=%d", devicenum);
}