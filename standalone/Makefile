NVCC = nvcc

NVCUFLAGS = 
ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi
endif


INCLUDE = -I include -I prims

# unit test source files
USRC = unittest/test_mpi.cu unittest/test_simple.cu unittest/test_ll.cu #unittest/test_ib.cu

BUILDDIR = build

OBJECTS = $(USRC:%.cu=$(BUILDDIR)/%.o)

EXE = $(USRC:%.cu=$(BUILDDIR)/%_perf)


all: $(EXE)

$(BUILDDIR)/unittest/%.o: unittest/%.cu
	mkdir -p $(BUILDDIR)/unittest
	$(NVCC) $(NVCUFLAGS) $(INCLUDE) -c $< -o $@

${BUILDDIR}/unittest/%_perf:${BUILDDIR}/unittest/%.o
	$(NVCC) $(NVLDFLAGS) $^ -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(EXE)
