#include <iostream>
#include <stdexcept>
#include <vector>
#include <queue>
#include <climits>
#include "graph.h"

using namespace std;

// CPU BFS implementation
void bfsCPU(int start, Graph &G, vector<int> &distance, vector<bool> &visited)
{
    // Initialize all distances to infinity (INT_MAX)
    fill(distance.begin(), distance.end(), INT_MAX);
    fill(visited.begin(), visited.end(), false);

    // Set starting vertex distance to 0
    distance[start] = 0;
    visited[start] = true;

    queue<int> to_visit;
    to_visit.push(start);

    while (!to_visit.empty())
    {
        int current = to_visit.front();
        to_visit.pop();

        // Iterate through all neighbors using CSR format
        int offset = G.edgesOffset[current];
        int numNeighbors = G.edgesSize[current];

        for (int i = 0; i < numNeighbors; ++i)
        {
            int neighbor = G.adjacencyList[offset + i];

            // If not visited, update distance and add to queue
            if (distance[neighbor] == INT_MAX)
            {
                distance[neighbor] = distance[current] + 1;
                visited[neighbor] = true;
                to_visit.push(neighbor);
            }
        }
    }
}

int main(int argc, char *argv[])
{
    // Check if CSV file path is provided as command line argument
    if (argc < 2)
    {
        cerr << "Usage: " << argv[0] << " <path_to_csv_file>" << endl;
        cerr << "Example: " << argv[0] << " ../graph_100_8.csv" << endl;
        return 1;
    }

    string csvFilePath = argv[1];

    try
    {
        // Create graph from CSV file
        Graph graph(csvFilePath);

        // Pretty print the graph
        graph.print();

        cout << "Graph successfully loaded and displayed!" << endl;

        // Run BFS from vertex 0
        int startVertex = 0;
        vector<int> distance(graph.numVertices);
        vector<bool> visited(graph.numVertices);

        cout << "\n============================================" << endl;
        cout << "Running BFS from vertex " << startVertex << endl;
        cout << "============================================" << endl;

        bfsCPU(startVertex, graph, distance, visited);

        // Display BFS results
        cout << "\nBFS Results:" << endl;
        cout << "--------------------------------------------" << endl;

        int numVisited = 0;
        int maxDistance = 0;

        for (int i = 0; i < graph.numVertices; ++i)
        {
            if (distance[i] != INT_MAX)
            {
                numVisited++;
                if (distance[i] > maxDistance)
                {
                    maxDistance = distance[i];
                }
            }
        }

        cout << "Vertices visited: " << numVisited << " / " << graph.numVertices << endl;
        cout << "Maximum distance: " << maxDistance << endl;
        cout << "--------------------------------------------" << endl;

        // Show first 20 vertices distances
        cout << "\nDistance from source (first 20 vertices):" << endl;
        for (int i = 0; i < min(20, graph.numVertices); ++i)
        {
            cout << "Vertex " << i << ": ";
            if (distance[i] == INT_MAX)
            {
                cout << "unreachable" << endl;
            }
            else
            {
                cout << distance[i] << endl;
            }
        }
        cout << endl;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}