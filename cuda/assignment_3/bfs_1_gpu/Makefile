# Compiler settings
NVCC = nvcc
CXX = g++
CXXFLAGS = -std=c++11 -O3
NVCCFLAGS = -std=c++11 -O3 -arch=sm_86

# Target executable
TARGET = graph_bfs

# Source files
CU_SOURCES = main.cu
CPP_SOURCES = graph.cpp
HEADERS = graph.h

# Object files
CU_OBJECTS = $(CU_SOURCES:.cu=.o)
CPP_OBJECTS = $(CPP_SOURCES:.cpp=.o)

# Default target
all: $(TARGET)

# Link object files
$(TARGET): $(CU_OBJECTS) $(CPP_OBJECTS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

# Compile .cu files
%.o: %.cu $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# Compile .cpp files
%.o: %.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Clean build artifacts
clean:
	rm -f $(TARGET) *.o

# Run with default graph
run: $(TARGET)
	./$(TARGET) ../graph_100_8.csv

.PHONY: all clean run
