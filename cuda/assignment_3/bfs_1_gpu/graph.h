#ifndef GRAPH_H
#define GRAPH_H

#include <vector>
#include <string>
#include <iostream>

// Class representing a graph using CSR (Compressed Sparse Row) format
class Graph
{
public:
    // Constructor that builds graph from CSV file
    Graph(const std::string &csvFilePath);

    // Graph data in CSR format
    std::vector<int> adjacencyList; // All neighbors stored consecutively
    std::vector<int> edgesOffset;   // Starting offset for each vertex's neighbors
    std::vector<int> edgesSize;     // Number of edges for each vertex
    int numVertices;
    int numEdges;

    // Feature vectors for Euclidean distance calculation
    std::vector<float> features; // Flattened feature vectors (numVertices * dimensions)
    int featureDimensions;       // Number of dimensions per feature vector

    // Load feature vectors from .fvecs file
    void loadFeatures(const std::string &fvecsFilePath);

    // Print graph information
    void print() const;

private:
    // Initialize graph from adjacency list representation
    void init(const std::vector<std::vector<int>> &adjList);
};

#endif // GRAPH_H
