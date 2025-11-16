# BFS CUDA Benchmark Suite

This directory contains a comprehensive benchmark suite for comparing three BFS implementations:

- **Baseline**: Standard CUDA implementation
- **Stream**: Optimized with separate transfer streams
- **CUDA Graph**: Stream capture mode with graph execution

## Files

- `benchmark.sh`: Main benchmark script
- `benchmark_results.txt`: Detailed results from the last run
- `benchmark_graphs/`: Directory containing generated test graphs

## Usage

```bash
./benchmark.sh
```

The script will:

1. Generate KNN graphs with varying sizes (nodes and degrees)
2. Run all three implementations on each graph
3. Record execution times and calculate speedups
4. Save results to `benchmark_results.txt`

## Test Configurations

The benchmark tests the following graph configurations:

| Nodes   | Degree | Graph Size |
| ------- | ------ | ---------- |
| 1,000   | 8      | 35 KB      |
| 5,000   | 8      | 216 KB     |
| 10,000  | 8      | 440 KB     |
| 50,000  | 8      | 2.6 MB     |
| 100,000 | 8      | 5.2 MB     |
| 10,000  | 16     | 821 KB     |
| 50,000  | 16     | 4.8 MB     |
| 100,000 | 16     | 9.7 MB     |
| 10,000  | 32     | 1.6 MB     |
| 50,000  | 32     | 9.2 MB     |

## Sample Results

From the most recent run:

**Best Performance Cases:**

- **100K nodes, 16 degree**: CUDA Graph 1.12x faster than baseline (55.2ms vs 62.1ms)
- **10K nodes, 16 degree**: Stream 1.15x faster than baseline (38.5ms vs 44.4ms)
- **1K nodes, 8 degree**: CUDA Graph 1.12x faster than baseline (44.5ms vs 50.2ms)

**Average Performance:**

- Baseline: 47.97 ms
- Stream: 48.54 ms (0.98x)
- CUDA Graph: 48.32 ms (0.99x)

## Key Observations

1. **CUDA Graph advantages appear with larger graphs**: Best speedup (1.12x) on 100K nodes with degree 16
2. **Stream optimization varies by workload**: Works best on medium-sized graphs (10K nodes)
3. **Small graphs show overhead**: Both optimizations have initialization overhead that impacts small graphs
4. **Degree matters**: Higher degrees (more neighbors) affect the relative performance of each approach

## Requirements

- CUDA 12.9 or compatible
- Python 3 with numpy and faiss libraries
- Compiled executables: `bfs_baseline`, `bfs_stream`, `bfs_graph`
- SIFT dataset: `../sift/sift_base.fvecs`

## Customization

To add more test configurations, edit the `TEST_CONFIGS` array in `benchmark.sh`:

```bash
TEST_CONFIGS=(
    "nodes degree"
    # e.g., "200000 16"
)
```

## Output Format

Results are saved in `benchmark_results.txt` with:

- System information (date, host, CUDA version)
- Per-test execution times
- Speedup calculations
- Summary statistics with averages
