# BFS Graph Builder from CSV

This project builds a graph data structure from a CSV file containing adjacency lists and displays it in a pretty format.

## Structure

The project consists of:

- **graph.h** - Graph class header with CSR (Compressed Sparse Row) format
- **graph.cpp** - Graph implementation with CSV parser
- **main.cu** - Main program that reads CSV and displays the graph
- **Makefile** - Build configuration

## Graph Format

The graph uses **CSR (Compressed Sparse Row)** format for efficient memory usage and GPU processing:

- `adjacencyList`: All neighbors stored consecutively
- `edgesOffset`: Starting index for each vertex's neighbors
- `edgesSize`: Number of neighbors for each vertex

## CSV Format

The CSV file should have:

- Header row: `node_id,n0,n1,n2,n3,...`
- Data rows: node_id followed by neighbor node IDs

Example:

```csv
node_id,n0,n1,n2,n3,n4,n5,n6,n7
0,2,6,74,21,18,20,8,9
1,3,14,7,30,11,9,58,19
2,0,6,74,16,18,21,20,8
```

## Building

```bash
make
```

This will compile the program and create the `graph_bfs` executable.

## Running

```bash
./graph_bfs <path_to_csv_file>
```

Example:

```bash
./graph_bfs ../graph_100_8.csv
```

Or use the shortcut:

```bash
make run
```

## Output

The program displays:

1. Graph statistics (number of vertices and edges)
2. Adjacency list representation for each vertex
3. CSR format details (offsets, sizes, and flattened adjacency list)

## Cleaning

```bash
make clean
```

This removes compiled binaries and object files.
