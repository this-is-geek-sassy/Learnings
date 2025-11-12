#include "graph.h"
#include <fstream>
#include <sstream>
#include <stdexcept>

using namespace std;

// Helper function to print vectors
void print_vector(const string &title, const vector<int> &v)
{
    cout << title << " { ";
    for (size_t i = 0; i < v.size(); ++i)
    {
        cout << v[i];
        if (i < v.size() - 1)
            cout << ", ";
    }
    cout << " }" << endl;
}

Graph::Graph(const string &csvFilePath)
{
    cout << "Started reading graph from: " << csvFilePath << endl;

    ifstream file(csvFilePath);
    if (!file.is_open())
    {
        throw runtime_error("Could not open file: " + csvFilePath);
    }

    vector<vector<int>> adjList;
    string line;

    // Skip header line
    if (!getline(file, line))
    {
        throw runtime_error("Empty CSV file");
    }

    // Read each line and parse adjacency list
    while (getline(file, line))
    {
        if (line.empty())
            continue;

        stringstream ss(line);
        string cell;
        vector<int> neighbors;

        // Skip the first column (node_id)
        getline(ss, cell, ',');

        // Read all neighbor nodes
        while (getline(ss, cell, ','))
        {
            if (!cell.empty())
            {
                neighbors.push_back(stoi(cell));
            }
        }

        adjList.push_back(neighbors);
    }

    file.close();

    if (adjList.empty())
    {
        throw runtime_error("No vertices found in CSV file");
    }

    // Initialize the graph structure
    init(adjList);

    cout << "Finished reading graph. Vertices: " << numVertices
         << ", Edges: " << numEdges << endl;
}

void Graph::init(const vector<vector<int>> &adjList)
{
    numVertices = adjList.size();
    numEdges = 0;

    // Build CSR format
    for (int i = 0; i < numVertices; ++i)
    {
        edgesOffset.push_back(adjacencyList.size());
        edgesSize.push_back(adjList[i].size());

        for (int neighbor : adjList[i])
        {
            adjacencyList.push_back(neighbor);
            ++numEdges;
        }
    }
}

void Graph::print() const
{
    printf("\n");
    printf("============================================\n");
    printf("           GRAPH STRUCTURE                  \n");
    printf("============================================\n");
    printf("Number of Vertices: %d\n", numVertices);
    printf("Number of Edges:    %d\n", numEdges);
    printf("============================================\n\n");

    // Print adjacency list for each vertex
    printf("Adjacency List Representation:\n");
    printf("--------------------------------------------\n");
    for (int i = 0; i < numVertices; ++i)
    {
        printf("Vertex %3d -> [ ", i);
        int offset = edgesOffset[i];
        int size = edgesSize[i];
        for (int j = 0; j < size; ++j)
        {
            printf("%d", adjacencyList[offset + j]);
            if (j < size - 1)
                printf(", ");
        }
        printf(" ]\n");
    }
    printf("--------------------------------------------\n\n");

    // Print CSR format details
    printf("CSR Format Details:\n");
    print_vector("Edges Offset:   ", edgesOffset);
    print_vector("Edges Size:     ", edgesSize);
    print_vector("Adjacency List: ", adjacencyList);
    printf("\n");
}
