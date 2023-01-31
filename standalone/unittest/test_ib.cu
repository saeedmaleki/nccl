#include "net_ib.h"

int main()
{
    setbuf(stdout, NULL);
    ncclIbInit();
}