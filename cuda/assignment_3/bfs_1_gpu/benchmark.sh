#!/bin/bash

# Benchmark script for BFS implementations
# Tests baseline, stream, and CUDA graph versions with varying graph sizes

set -e

# Output file for results
RESULTS_FILE="benchmark_results.txt"
GRAPHS_DIR="benchmark_graphs"
SIFT_FILE="../sift/sift_base.fvecs"

# Create directory for benchmark graphs
mkdir -p "$GRAPHS_DIR"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  BFS CUDA Implementation Benchmark${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if executables exist
if [ ! -f "./bfs_baseline" ] || [ ! -f "./bfs_stream" ] || [ ! -f "./bfs_graph" ]; then
    echo -e "${YELLOW}Building all executables...${NC}"
    make clean
    make all
fi

# Initialize results file
echo "BFS CUDA Implementation Benchmark Results" > "$RESULTS_FILE"
echo "=========================================" >> "$RESULTS_FILE"
echo "Date: $(date)" >> "$RESULTS_FILE"
echo "Host: $(hostname)" >> "$RESULTS_FILE"
echo "CUDA Version: $(nvcc --version | grep release)" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Test configurations: (nodes, degree)
# Small to large graphs
TEST_CONFIGS=(
    "1000 8"
    "5000 8"
    "10000 8"
    "50000 8"
    "100000 8"
    "10000 16"
    "50000 16"
    "100000 16"
    "10000 32"
    "50000 32"
)

echo -e "${GREEN}Generating test graphs...${NC}"
echo ""

# Generate all graphs first
for config in "${TEST_CONFIGS[@]}"; do
    read -r nodes degree <<< "$config"
    graph_file="${GRAPHS_DIR}/graph_${nodes}_${degree}.csv"
    
    if [ ! -f "$graph_file" ]; then
        echo -e "${YELLOW}Creating graph: N=${nodes}, R=${degree}${NC}"
        python3 ../build_knn_graph.py "$SIFT_FILE" -N "$nodes" -R "$degree" -o "$graph_file" > "${graph_file}.output.txt" 2>&1
        start_node=$(grep "Node ID:" "${graph_file}.output.txt" | awk '{print $3}')
        echo "$start_node" > "${graph_file}.startnode"
        echo "  Graph saved to $graph_file"
        echo "  Start node: $start_node"
        echo ""
    fi
done

echo -e "${GREEN}Starting benchmarks...${NC}"
echo ""
echo "=========================================" >> "$RESULTS_FILE"
echo "BENCHMARK RESULTS" >> "$RESULTS_FILE"
echo "=========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Function to extract execution time from output
extract_time() {
    local output="$1"
    local impl_type="$2"
    
    if [ "$impl_type" == "baseline" ]; then
        echo "$output" | grep "GPU Execution Time:" | grep -oP '\d+\.\d+' || echo "ERROR"
    elif [ "$impl_type" == "stream" ]; then
        echo "$output" | grep "GPU Execution Time (Stream):" | grep -oP '\d+\.\d+' || echo "ERROR"
    elif [ "$impl_type" == "graph" ]; then
        echo "$output" | grep "GPU Execution Time (CUDA Graph):" | grep -oP '\d+\.\d+' || echo "ERROR"
    fi
}

# Run benchmarks
run_count=0
total_tests=$((${#TEST_CONFIGS[@]} * 3))

for config in "${TEST_CONFIGS[@]}"; do
    read -r nodes degree <<< "$config"
    graph_file="${GRAPHS_DIR}/graph_${nodes}_${degree}.csv"
    start_node=$(cat "${graph_file}.startnode")
    
    echo -e "${BLUE}----------------------------------------${NC}"
    echo -e "${BLUE}Testing: N=${nodes}, R=${degree}, Start=${start_node}${NC}"
    echo -e "${BLUE}----------------------------------------${NC}"
    
    # Log to results file
    echo "" >> "$RESULTS_FILE"
    echo "Test Configuration: Nodes=$nodes, Degree=$degree" >> "$RESULTS_FILE"
    echo "Start Node: $start_node" >> "$RESULTS_FILE"
    echo "----------------------------------------" >> "$RESULTS_FILE"
    
    # Test Baseline
    run_count=$((run_count + 1))
    echo -e "${YELLOW}[$run_count/$total_tests] Running baseline...${NC}"
    baseline_output=$(./bfs_baseline "$graph_file" "$start_node" "$SIFT_FILE" 2>&1)
    baseline_time=$(extract_time "$baseline_output" "baseline")
    echo "  Time: ${baseline_time} ms"
    echo "Baseline Time: ${baseline_time} ms" >> "$RESULTS_FILE"
    
    # Test Stream
    run_count=$((run_count + 1))
    echo -e "${YELLOW}[$run_count/$total_tests] Running stream...${NC}"
    stream_output=$(./bfs_stream "$graph_file" "$start_node" "$SIFT_FILE" 2>&1)
    stream_time=$(extract_time "$stream_output" "stream")
    echo "  Time: ${stream_time} ms"
    echo "Stream Time: ${stream_time} ms" >> "$RESULTS_FILE"
    
    # Test CUDA Graph
    run_count=$((run_count + 1))
    echo -e "${YELLOW}[$run_count/$total_tests] Running CUDA graph...${NC}"
    graph_output=$(./bfs_graph "$graph_file" "$start_node" "$SIFT_FILE" 2>&1)
    graph_time=$(extract_time "$graph_output" "graph")
    echo "  Time: ${graph_time} ms"
    echo "CUDA Graph Time: ${graph_time} ms" >> "$RESULTS_FILE"
    
    # Calculate speedups
    if [ "$baseline_time" != "ERROR" ] && [ "$stream_time" != "ERROR" ] && [ "$graph_time" != "ERROR" ]; then
        stream_speedup=$(echo "scale=2; $baseline_time / $stream_time" | bc)
        graph_speedup=$(echo "scale=2; $baseline_time / $graph_time" | bc)
        
        echo "" >> "$RESULTS_FILE"
        echo "Stream Speedup: ${stream_speedup}x" >> "$RESULTS_FILE"
        echo "CUDA Graph Speedup: ${graph_speedup}x" >> "$RESULTS_FILE"
        
        echo -e "${GREEN}  Stream speedup: ${stream_speedup}x${NC}"
        echo -e "${GREEN}  CUDA Graph speedup: ${graph_speedup}x${NC}"
    fi
    
    echo ""
done

# Generate summary statistics
echo "" >> "$RESULTS_FILE"
echo "=========================================" >> "$RESULTS_FILE"
echo "SUMMARY" >> "$RESULTS_FILE"
echo "=========================================" >> "$RESULTS_FILE"
echo "" >> "$RESULTS_FILE"

# Extract all times and calculate averages
baseline_times=$(grep "Baseline Time:" "$RESULTS_FILE" | grep -oP '\d+\.\d+')
stream_times=$(grep "Stream Time:" "$RESULTS_FILE" | grep -oP '\d+\.\d+')
graph_times=$(grep "CUDA Graph Time:" "$RESULTS_FILE" | grep -oP '\d+\.\d+')

if [ -n "$baseline_times" ]; then
    avg_baseline=$(echo "$baseline_times" | awk '{sum+=$1; count++} END {print sum/count}')
    avg_stream=$(echo "$stream_times" | awk '{sum+=$1; count++} END {print sum/count}')
    avg_graph=$(echo "$graph_times" | awk '{sum+=$1; count++} END {print sum/count}')
    
    echo "Average Execution Times:" >> "$RESULTS_FILE"
    echo "  Baseline: ${avg_baseline} ms" >> "$RESULTS_FILE"
    echo "  Stream: ${avg_stream} ms" >> "$RESULTS_FILE"
    echo "  CUDA Graph: ${avg_graph} ms" >> "$RESULTS_FILE"
    echo "" >> "$RESULTS_FILE"
    
    avg_stream_speedup=$(echo "scale=2; $avg_baseline / $avg_stream" | bc)
    avg_graph_speedup=$(echo "scale=2; $avg_baseline / $avg_graph" | bc)
    
    echo "Average Speedups:" >> "$RESULTS_FILE"
    echo "  Stream vs Baseline: ${avg_stream_speedup}x" >> "$RESULTS_FILE"
    echo "  CUDA Graph vs Baseline: ${avg_graph_speedup}x" >> "$RESULTS_FILE"
    
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Benchmark Complete!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo -e "Average Speedups:"
    echo -e "  Stream: ${GREEN}${avg_stream_speedup}x${NC}"
    echo -e "  CUDA Graph: ${GREEN}${avg_graph_speedup}x${NC}"
fi

echo ""
echo -e "${BLUE}Results saved to: ${RESULTS_FILE}${NC}"
echo -e "${BLUE}Benchmark graphs in: ${GRAPHS_DIR}/${NC}"
echo ""
