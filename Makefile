NVCC=nvcc
LDFLAGS=-Xcompiler="-pthread" -lcurand
CUDAFLAGS=-dc 
SOURCES=common.cu sha3.cu main.cu
OBJECTS=$(SOURCES:.cu=.o)
EXECUTABLE=kecmatch64_linux

all: $(EXECUTABLE)

$(EXECUTABLE): $(OBJECTS)
	$(NVCC) $(OBJECTS) $(LDFLAGS) -o $@

%.o:    %.cu
	$(NVCC) $(CUDAFLAGS) $< -o $@

clean:
	rm -f test *.o $(EXECUTABLE) 
