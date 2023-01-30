NVCC = nvcc

NVCUFLAGS = 
ifeq ($(MPI), 1)
NVCUFLAGS += -DMPI_SUPPORT -I$(MPI_HOME)/include
NVLDFLAGS += -L$(MPI_HOME)/lib -L$(MPI_HOME)/lib64 -lmpi
endif


INCLUDE = -I include -I prims

# unit test source files
UNITTESTSRC = unittest/test_mpi.cu unittest/test_simple.cu unittest/test_ll.cu unittest/test_ib.cu
# UNITTESTSRC = unittest/test_simple.cu
LIBSRC = debug.cc misc/ibvwrap.cc socket.cc

BUILDDIR = build

OBJECTS = $(UNITTESTSRC:%.cu=$(BUILDDIR)/%.o)

EXE = $(UNITTESTSRC:%.cu=$(BUILDDIR)/%_perf)

LIBOBJ = $(LIBSRC:%.cc=$(BUILDDIR)/%.o)

all: $(EXE)

$(BUILDDIR)/unittest/%.o: unittest/%.cu 
	mkdir -p $(BUILDDIR)/unittest
	$(NVCC) $(NVCUFLAGS) $(INCLUDE) -c $< -o $@

# $(BUILDDIR)/%.o: %.cc
# 	mkdir -p $(BUILDDIR)
# 	$(NVCC) $(NVCUFLAGS) $(INCLUDE) -c $< -o $@

${BUILDDIR}/unittest/%_perf:${BUILDDIR}/unittest/%.o
	$(NVCC) $(NVLDFLAGS) $^ -o $@



# Clean up
clean:
	rm -f $(OBJECTS) $(EXE)
