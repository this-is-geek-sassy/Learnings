# Compiler settings
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -O3
NVCCFLAGS = -std=c++11 -O3 -arch=sm_86

# Target executables
TARGET_BASELINE = bfs_baseline
TARGET_STREAM = bfs_stream
TARGET_GRAPH = bfs_graph

# Source files
CU_BASELINE = bfs_baseline.cu
CU_STREAM = bfs_stream.cu
CU_GRAPH = bfs_graph.cu
CPP_SOURCES = graph.cpp
HEADERS = graph.h

# Object files
OBJ_BASELINE = bfs_baseline.o
OBJ_STREAM = bfs_stream.o
OBJ_CUDAGRAPH = bfs_graph.o
OBJ_GRAPH = graph.o

# Default target - build all versions
all: $(TARGET_BASELINE) $(TARGET_STREAM) $(TARGET_GRAPH)

# Link baseline version
$(TARGET_BASELINE): $(OBJ_BASELINE) $(OBJ_GRAPH)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Link stream version
$(TARGET_STREAM): $(OBJ_STREAM) $(OBJ_GRAPH)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Link CUDA graph version
$(TARGET_GRAPH): $(OBJ_CUDAGRAPH) $(OBJ_GRAPH)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile .cu files
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile .cpp files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(TARGET_BASELINE) $(TARGET_STREAM) $(TARGET_GRAPH) *.o

# Run baseline with default graph
run-baseline: $(TARGET_BASELINE)
	./$(TARGET_BASELINE) ../graph_100_8.csv

# Run stream version with default graph
run-stream: $(TARGET_STREAM)
	./$(TARGET_STREAM) ../graph_100_8.csv

# Run CUDA graph version with default graph
run-graph: $(TARGET_GRAPH)
	./$(TARGET_GRAPH) ../graph_100_8.csv
	./$(TARGET_STREAM) ../graph_100_8.csv

.PHONY: all clean run-baseline run-stream
