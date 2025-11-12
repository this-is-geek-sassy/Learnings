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

    // Print graph information
    void print() const;

private:
    // Initialize graph from adjacency list representation
    void init(const std::vector<std::vector<int>> &adjList);
};

#endif // GRAPH_H
