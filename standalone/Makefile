CC = nvcc

CFLAGS = 

INCLUDE = -I include -I prims

# Source files
SOURCES = test.cu

BUILDDIR = build

OBJECTS = $(BUILDDIR)/$(SOURCES:.cu=.o)

EXE = $(BUILDDIR)/$(SOURCES:.cu=)

all: $(EXE)

$(OBJECTS): $(SOURCES)
	mkdir -p $(BUILDDIR)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@

$(EXE): $(OBJECTS)
	$(CC) $(CFLAGS) $^ -o $@

# Clean up
clean:
	rm -f $(OBJECTS) $(EXE)
