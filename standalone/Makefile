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
LIBSRC = debug.cc misc/ibvwrap.cc misc/socket.cc misc/utils.cc misc/nvmlwrap.cc misc/param.cc

BUILDDIR = build


UNITTEST = $(UNITTESTSRC:%.cu=$(BUILDDIR)/%_perf)

LIBOBJ := $(LIBSRC:%.cc=$(BUILDDIR)/%.o)
UNITTESTOBJ := $(UNITTESTSRC:%.cu=$(BUILDDIR)/%.o)

all: $(UNITTEST)

$(BUILDDIR)/unittest/%.o: unittest/%.cu 
	mkdir -p $(BUILDDIR)/unittest
	$(NVCC) $(NVCUFLAGS) $(INCLUDE) -c $< -o $@

$(LIBOBJ): $(BUILDDIR)/%.o: %.cc
	mkdir -p $(BUILDDIR)/misc
	$(NVCC) $(NVCUFLAGS) $(INCLUDE) -c $< -o $@

${UNITTEST}: ${BUILDDIR}/unittest/%_perf:${BUILDDIR}/unittest/%.o $(LIBOBJ)
	$(NVCC) $(NVLDFLAGS) $^ -o $@

# Clean up
clean:
	rm -r $(BUILDDIR)/*
