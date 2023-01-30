NVCC = nvcc

NVCUFLAGS = 
ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi
endif


INCLUDE = -I include -I prims

# Source files
SOURCES := test_mpi.cu test_single.cu test_ll.cu


BUILDDIR = build

OBJECTS = $(SOURCES:%.cu=$(BUILDDIR)/%.o)

EXE = $(BUILDDIR)/$(SOURCES:.cu=_perf)


all: $(EXE)

$(BUILDDIR)/%.o: %.cu
	mkdir -p $(BUILDDIR)
	$(NVCC) $(NVCUFLAGS) $(INCLUDE) -c $< -o $@

${BUILDDIR}/%_perf:${BUILDDIR}/%.o
	$(NVCC) $(NVLDFLAGS) $^ -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(EXE)
