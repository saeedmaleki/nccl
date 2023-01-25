NVCC = nvcc

CFLAGS = 

INCLUDE = -I include -I prims

# Source files
SOURCES = test_simple.cu test_ll.cu

BUILDDIR = build

OBJECTS = $(BUILDDIR)/test_simple.o $(BUILDDIR)/test_ll.o

EXE = $(BUILDDIR)/test_simple $(BUILDDIR)/test_ll

all: $(EXE)

$(BUILDDIR)/test_simple: test_simple.cu
	mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) $(INCLUDE) test_simple.cu -o $@

$(BUILDDIR)/test_ll: test_ll.cu
	mkdir -p $(BUILDDIR)
	$(NVCC) $(CFLAGS) $(INCLUDE) test_ll.cu -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(EXE)
