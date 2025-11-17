# CUDA-Accelerated Breadth-First Search on KNN Graphs

This project implements three progressively optimized versions of GPU-accelerated Breadth-First Search (BFS) on K-Nearest Neighbor (KNN) graphs, with comprehensive benchmarking and performance analysis.

## Project Overview

The project explores different CUDA optimization techniques for BFS traversal:

1. **Baseline Implementation** - Standard GPU BFS with full adjacency list transfer
2. **Stream-Optimized** - Uses multiple CUDA streams for concurrent data transfers
3. **CUDA Graph** - Leverages CUDA Graphs with stream capture for maximum performance

## Project Structure

```
bfs_1_gpu/
├── src/
│   ├── graph/
│   │   ├── graph.h           # Graph class with CSR format
│   │   └── graph.cpp          # CSV parser & SIFT feature loader
│   ├── cpu/
│   │   ├── bfs.h              # CPU BFS header
│   │   └── bfs.cpp            # CPU BFS implementation
│   └── gpu/
│       ├── bfs_baseline.cu    # Standard GPU BFS implementation
│       ├── bfs_stream.cu      # Stream-optimized version
│       └── bfs_graph.cu       # CUDA Graph implementation
├── Makefile                   # Build configuration for all versions
├── benchmark.sh               # Automated benchmark script
├── plot_benchmark.py          # Performance visualization script
├── build_knn_graph.py         # KNN graph generator
├── report.tex                 # Comprehensive LaTeX report
└── README.md                  # This file
```

## Data Structures

### Graph Representation (CSR Format)

The graph uses **Compressed Sparse Row (CSR)** format for efficient GPU processing:

- `adjacencyList`: All neighbors stored consecutively in a flat array
- `edgesOffset`: Starting index in adjacencyList for each vertex's neighbors
- `edgesSize`: Number of neighbors for each vertex

### Feature Vectors (SIFT Dataset)

- **128-dimensional SIFT descriptors** from `sift_base.fvecs`
- Used for computing Euclidean distances between graph nodes
- Provides geometric meaning to graph structure

### Input Format (CSV)

KNN graph CSV files should have:

- Header: `node_id,n0,n1,n2,n3,...`
- Data rows: node_id followed by K neighbor IDs

Example:

```csv
node_id,n0,n1,n2,n3,n4,n5,n6,n7
0,2,6,74,21,18,20,8,9
1,3,14,7,30,11,9,58,19
```

## Building

### Build All Implementations

```bash
make
```

Creates three executables:

- `bfs_baseline` - Standard GPU BFS
- `bfs_stream` - Stream-optimized version
- `bfs_graph` - CUDA Graph implementation

### Build Individual Versions

```bash
make bfs_baseline    # Build baseline only
make bfs_stream      # Build stream version only
make bfs_graph       # Build CUDA Graph version only
```

### Clean Build Artifacts

```bash
make clean           # Remove all binaries and object files
```

## Running Individual Programs

All programs require two command-line arguments:

1. Path to KNN graph CSV file
2. Path to SIFT feature vectors file (`.fvecs` format)

### Baseline Implementation

```bash
./bfs_baseline <graph_csv> <features_fvecs>
```

Example:

```bash
./bfs_baseline ../graph_100_8.csv ../sift/sift_base.fvecs
```

### Stream-Optimized Version

```bash
./bfs_stream <graph_csv> <features_fvecs>
```

Example:

```bash
./bfs_stream ../graph_100_8.csv ../sift/sift_base.fvecs
```

### CUDA Graph Version

```bash
./bfs_graph <graph_csv> <features_fvecs>
```

Example:

```bash
./bfs_graph ../graph_100_8.csv ../sift/sift_base.fvecs
```

## Program Output

Each program outputs:

1. **Graph Statistics**: Number of vertices and edges
2. **BFS Traversal Results**:
   - Level-by-level node traversal from source vertex 0
   - Euclidean distances from source (using SIFT features)
   - Format: `Node <id> at level <level>, distance: <euclidean_dist>`
3. **Verification**: CPU vs GPU correctness check
4. **Performance Metrics**:
   - GPU execution time (in milliseconds)
   - Total nodes visited and edges explored
5. **Output Files**:
   - `output_<version>.txt` - Complete traversal results
   - Example: `output_baseline.txt`, `output_stream.txt`, `output_graph.txt`

## Benchmarking

### Automated Benchmark Suite

The `benchmark.sh` script tests all three implementations across multiple graph configurations:

```bash
./benchmark.sh
```

**What it does**:

1. Generates 10 different KNN graphs:
   - Sizes: 1,000 / 5,000 / 10,000 / 50,000 / 100,000 nodes
   - Degrees (K): 8, 16, 32 neighbors per node
2. Runs each implementation on every graph
3. Extracts execution times and calculates speedups
4. Saves results to `benchmark_results.txt`

**Generated Graphs**:

- `graph_1000_8.csv`, `graph_1000_16.csv`, `graph_1000_32.csv`
- `graph_5000_8.csv`, `graph_5000_16.csv`
- `graph_10000_8.csv`, `graph_10000_16.csv`
- `graph_50000_8.csv`, `graph_50000_32.csv`
- `graph_100000_16.csv`

**Benchmark Output Format**:

```
=== Graph: 10000 nodes, degree 8 ===
Baseline: 15.234 ms
Stream: 14.876 ms (speedup: 1.02x)
Graph: 13.542 ms (speedup: 1.12x)
```

### Manual Benchmarking

To benchmark a specific graph:

```bash
# Run each version and compare times
./bfs_baseline graph.csv sift_base.fvecs | grep "GPU Time"
./bfs_stream graph.csv sift_base.fvecs | grep "GPU Time"
./bfs_graph graph.csv sift_base.fvecs | grep "GPU Time"
```

## Performance Visualization

### Generate Performance Plots

```bash
python3 plot_benchmark.py
```

**Requirements**:

```bash
pip install matplotlib numpy
```

**Output**:

- `benchmark_plots.png` - 6-subplot comparison chart:
  - Top row: Runtime vs number of nodes (for degree 8, 16, 32)
  - Bottom row: Runtime vs degree (for 1K, 10K, 100K nodes)
  - Color-coded lines for baseline/stream/graph versions

## Generating Custom Graphs

### Using the Graph Generator

```bash
python3 build_knn_graph.py <num_nodes> <k_neighbors> <output_csv>
```

**Parameters**:

- `num_nodes`: Number of vertices in the graph
- `k_neighbors`: Number of neighbors per node (K in KNN)
- `output_csv`: Output filename for the generated graph

**Examples**:

```bash
# Generate 1000-node graph with 8 neighbors each
python3 build_knn_graph.py 1000 8 graph_1000_8.csv

# Generate 50000-node graph with 16 neighbors each
python3 build_knn_graph.py 50000 16 graph_50000_16.csv

# Generate large graph for stress testing
python3 build_knn_graph.py 100000 32 graph_100k_32.csv
```

**How it works**:

- Uses SIFT feature vectors from `sift_base.fvecs`
- Computes K-nearest neighbors based on Euclidean distance
- Generates bidirectional edges for graph connectivity
- Outputs CSV in the required format

## Implementation Details

### Baseline Implementation (`bfs_baseline.cu`)

- **Strategy**: Full adjacency list transfer to GPU
- **Memory**: Allocates device memory for entire graph structure
- **Execution**: Single kernel launch per BFS level
- **Best for**: Small to medium graphs, simplicity

### Stream-Optimized (`bfs_stream.cu`)

- **Strategy**: Uses 3 separate CUDA streams for concurrent operations
  - `transferStream1`: Transfers neighbor data
  - `transferStream2`: Transfers edge offsets
  - `transferStream3`: Transfers edge sizes
- **Memory**: Pinned host memory for faster DMA transfers
- **Execution**: Overlaps data transfer with computation
- **Best for**: Graphs with high transfer overhead

### CUDA Graph Implementation (`bfs_graph.cu`)

- **Strategy**: Captures entire BFS workflow as CUDA Graph
- **Features**:
  - Stream capture mode (`cudaStreamBeginCapture`)
  - Asynchronous memory operations (`cudaMemcpyAsync`)
  - Dynamic parameter updates (`cudaGraphExecKernelNodeSetParams`)
- **Memory**: Pinned host memory with reusable graph structure
- **Execution**: Single graph launch per BFS level with parameter updates
- **Best for**: Large graphs with repeated operations, lowest overhead

### Performance Characteristics

From comprehensive benchmarking:

- **CUDA Graph**: 15-20% faster than baseline (best for 50K+ nodes)
- **Stream**: 5-15% faster than baseline (best for medium graphs)
- **Baseline**: Reference implementation (simplest, good for small graphs)

## Troubleshooting

### CUDA Compute Capability

If compilation fails, check your GPU compute capability:

```bash
nvidia-smi --query-gpu=compute_cap --format=csv
```

Update `Makefile` line 2:

```makefile
ARCH = -arch=sm_86  # Change to your GPU's compute capability
```

### Memory Issues

For very large graphs (>100K nodes):

- Check available GPU memory: `nvidia-smi`
- Reduce graph size or K value
- Use stream version for better memory management

### Missing SIFT Data

Ensure `sift_base.fvecs` is available:

```bash
ls ../sift/sift_base.fvecs
```

Download from [SIFT1M dataset](http://corpus-texmex.irisa.fr/) if missing.

## Documentation

Full technical report available in `report.tex`:

```bash
pdflatex report.tex
```

The report includes:

- Detailed algorithm descriptions
- Implementation strategies
- Comprehensive benchmark results
- Performance analysis and comparisons
- Optimization techniques discussion

## License

This project is for educational purposes as part of GPU Programming coursework at IIT Hyderabad.
