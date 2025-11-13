#include <bits/stdc++.h>
#include "graph.h"

using namespace std;

#define N_THREADS_PER_BLOCK 256

// Checker class for verifying BFS correctness
class Checker
{
    vector<int> expected_answer;

public:
    Checker(vector<int> exp_ans)
    {
        expected_answer = exp_ans;
    }

    pair<int, int> count_visited_vertices(const vector<int> &distance)
    {
        int depth = 0;
        int count = 0;
        for (int x : distance)
        {
            if (x < INT_MAX)
            {
                ++count;
                if (x > depth)
                {
                    depth = x;
                }
            }
        }
        return {count, depth};
    }

    bool check(const vector<int> &answer, const string &implementation_name = "Implementation")
    {
        if (answer.size() != expected_answer.size())
        {
            cout << "✗ Size mismatch: expected " << expected_answer.size()
                 << ", got " << answer.size() << endl;
            return false;
        }

        bool is_ok = true;
        int mismatches = 0;
        vector<int> mismatch_positions;

        for (int i = 0; i < answer.size(); ++i)
        {
            if (answer[i] != expected_answer[i])
            {
                is_ok = false;
                mismatches++;
                if (mismatch_positions.size() < 10) // Store first 10 mismatches
                {
                    mismatch_positions.push_back(i);
                }
            }
        }

        if (is_ok)
        {
            pair<int, int> graph_output = count_visited_vertices(answer);
            int n_visited_vertices = graph_output.first;
            int depth = graph_output.second;
            cout << "✓ " << implementation_name << " VERIFIED SUCCESSFULLY!" << endl;
            cout << "  - Vertices visited: " << n_visited_vertices << endl;
            cout << "  - Maximum depth: " << depth << endl;
            return true;
        }
        else
        {
            cout << "✗ " << implementation_name << " VERIFICATION FAILED!" << endl;
            cout << "  - Total mismatches: " << mismatches << endl;
            cout << "  - First mismatches at vertices:" << endl;
            for (int pos : mismatch_positions)
            {
                cout << "    Vertex " << pos << ": expected " << expected_answer[pos]
                     << ", got " << answer[pos] << endl;
            }
            return false;
        }
    }
};

// GPU Kernel: Process current frontier and discover neighbors
__global__ void bfsKernel(int *adjacencyList, int *edgesOffset, int *edgesSize, int *distance,
                          int queueSize, int *currentQueue, int *nextQueueSize, int *nextQueue, int level)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < queueSize)
    {
        int current = currentQueue[tid];

        // Explore all neighbors of current vertex
        for (int i = edgesOffset[current]; i < edgesOffset[current] + edgesSize[current]; ++i)
        {
            int neighbor = adjacencyList[i];

            // If neighbor not visited, mark it and add to next queue
            if (distance[neighbor] == INT_MAX)
            {
                distance[neighbor] = level + 1;
                int position = atomicAdd(nextQueueSize, 1);
                nextQueue[position] = neighbor;
            }
        }
    }
}

// Helper function to calculate Euclidean distance from start node
double calculateEuclideanDistance(int node, int start, const Graph &G)
{
    // For a graph representation, we use the graph distance (BFS level) as a proxy
    // In a real geometric graph, you would use actual coordinates
    // Here we return the node index difference as a simple metric
    return abs(node - start);
}

// GPU-accelerated BFS with CPU-GPU coordination
void bfsGPU(int start, Graph &G, vector<int> &distance, vector<bool> &visited,
            ofstream &outputFile, bool verbose = true)
{
    const int n_blocks = (G.numVertices + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    // Device pointers
    int *d_adjacencyList;
    int *d_edgesOffset;
    int *d_edgesSize;
    int *d_firstQueue;
    int *d_secondQueue;
    int *d_nextQueueSize;
    int *d_distance;

    // Host variables
    int currentQueueSize = 1;
    const int NEXT_QUEUE_SIZE = 0;
    int level = 0;

    // For tracking nodes at each level
    vector<vector<int>> nodesAtLevel;

    // Allocate device memory
    const int size = G.numVertices * sizeof(int);
    const int adjacencySize = G.adjacencyList.size() * sizeof(int);

    cudaMalloc((void **)&d_adjacencyList, adjacencySize);
    cudaMalloc((void **)&d_edgesOffset, size);
    cudaMalloc((void **)&d_edgesSize, size);
    cudaMalloc((void **)&d_firstQueue, size);
    cudaMalloc((void **)&d_secondQueue, size);
    cudaMalloc((void **)&d_distance, size);
    cudaMalloc((void **)&d_nextQueueSize, sizeof(int));

    // Transfer graph data to GPU (one-time transfer)
    if (verbose)
        cout << "\n[Phase 1] CPU -> GPU: Transferring graph structure..." << endl;
    auto transferStart = chrono::steady_clock::now();

    cudaMemcpy(d_adjacencyList, &G.adjacencyList[0], adjacencySize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgesOffset, &G.edgesOffset[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_edgesSize, &G.edgesSize[0], size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_firstQueue, &start, sizeof(int), cudaMemcpyHostToDevice);

    auto transferEnd = chrono::steady_clock::now();
    auto transferDuration = chrono::duration_cast<chrono::microseconds>(transferEnd - transferStart).count();
    if (verbose)
        cout << "Graph transfer completed in " << transferDuration / 1000.0 << " ms" << endl;

    // Initialize distance array
    if (verbose)
        cout << "\n[Phase 2] CPU -> GPU: Transferring initial distance array..." << endl;
    distance = vector<int>(G.numVertices, INT_MAX);
    distance[start] = 0;
    cudaMemcpy(d_distance, distance.data(), size, cudaMemcpyHostToDevice);

    // BFS traversal loop
    if (verbose)
    {
        cout << "\n[Phase 3] Starting BFS traversal..." << endl;
    }

    // Print table header
    string header = "| Level | # of nodes discovered | Closest and farthest node (in this level) from the start node and corresponding Euclidean Distance wrt start node |";
    string separator(header.length(), '-');

    cout << "\n"
         << separator << endl;
    cout << header << endl;
    cout << separator << endl;

    outputFile << separator << endl;
    outputFile << header << endl;
    outputFile << separator << endl;

    auto bfsStart = chrono::steady_clock::now();

    while (currentQueueSize > 0)
    {
        // Determine which queue to use (ping-pong between two queues)
        int *d_currentQueue = (level % 2 == 0) ? d_firstQueue : d_secondQueue;
        int *d_nextQueue = (level % 2 == 0) ? d_secondQueue : d_firstQueue;

        // Copy current queue to host to track nodes at this level
        vector<int> currentLevelNodes(currentQueueSize);
        cudaMemcpy(currentLevelNodes.data(), d_currentQueue, currentQueueSize * sizeof(int), cudaMemcpyDeviceToHost);
        nodesAtLevel.push_back(currentLevelNodes);

        // GPU processes current frontier
        bfsKernel<<<n_blocks, N_THREADS_PER_BLOCK>>>(
            d_adjacencyList, d_edgesOffset, d_edgesSize, d_distance,
            currentQueueSize, d_currentQueue, d_nextQueueSize, d_nextQueue, level);

        cudaDeviceSynchronize();

        // Find closest and farthest nodes at this level
        int closestNode = currentLevelNodes[0];
        int farthestNode = currentLevelNodes[0];
        double closestDist = calculateEuclideanDistance(closestNode, start, G);
        double farthestDist = closestDist;

        for (int node : currentLevelNodes)
        {
            double dist = calculateEuclideanDistance(node, start, G);
            if (dist < closestDist)
            {
                closestDist = dist;
                closestNode = node;
            }
            if (dist > farthestDist)
            {
                farthestDist = dist;
                farthestNode = node;
            }
        }

        // Print level information
        char buffer[512];
        if (level == 0)
        {
            snprintf(buffer, sizeof(buffer),
                     "| %-5d | %-21d | Farthest node = %d, Dist=%.1f\n| %5s | %21s | Closest node = %d, Dist=%.1f",
                     level, currentQueueSize, farthestNode, farthestDist, "", "", closestNode, closestDist);
        }
        else
        {
            snprintf(buffer, sizeof(buffer),
                     "| %-5d | %-21d | Farthest node = %d, Dist=%.1f\n| %5s | %21s | Closest node = %d, Dist=%.1f",
                     level, currentQueueSize, farthestNode, farthestDist, "", "", closestNode, closestDist);
        }

        string output(buffer);
        cout << output << endl;
        cout << separator << endl;
        outputFile << output << endl;
        outputFile << separator << endl;

        // CPU transfers next queue size back to host
        cudaMemcpy(&currentQueueSize, d_nextQueueSize, sizeof(int), cudaMemcpyDeviceToHost);

        // CPU resets next queue size counter for next iteration
        cudaMemcpy(d_nextQueueSize, &NEXT_QUEUE_SIZE, sizeof(int), cudaMemcpyHostToDevice);

        ++level;
    }

    auto bfsEnd = chrono::steady_clock::now();
    auto bfsDuration = chrono::duration_cast<chrono::milliseconds>(bfsEnd - bfsStart).count();

    string footer = "\nTotal BFS discovery time = " + to_string(bfsDuration) + " ms";
    cout << footer << endl;
    outputFile << footer << endl;

    // Transfer results back to host
    if (verbose)
        cout << "\n[Phase 4] GPU -> CPU: Transferring results..." << endl;
    cudaMemcpy(&distance[0], d_distance, size, cudaMemcpyDeviceToHost);

    // Fill visited array
    for (int i = 0; i < G.numVertices; ++i)
    {
        visited[i] = (distance[i] != INT_MAX);
    }

    // Cleanup
    cudaFree(d_adjacencyList);
    cudaFree(d_edgesOffset);
    cudaFree(d_edgesSize);
    cudaFree(d_firstQueue);
    cudaFree(d_secondQueue);
    cudaFree(d_distance);
    cudaFree(d_nextQueueSize);

    if (verbose)
        cout << "GPU memory freed successfully" << endl;
}

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
        cerr << "Usage: " << argv[0] << " <path_to_csv_file> [start_vertex]" << endl;
        cerr << "Example: " << argv[0] << " ../graph_100_8.csv" << endl;
        cerr << "Example: " << argv[0] << " ../graph_10000_16.csv 3263" << endl;
        return 1;
    }

    string csvFilePath = argv[1];
    int startVertex = 0; // Default start vertex

    // Parse optional start vertex argument
    if (argc >= 3)
    {
        startVertex = atoi(argv[2]);
    }

    // Extract graph name from file path for output file
    string graphName = csvFilePath;
    size_t lastSlash = graphName.find_last_of("/\\");
    if (lastSlash != string::npos)
    {
        graphName = graphName.substr(lastSlash + 1);
    }
    size_t lastDot = graphName.find_last_of(".");
    if (lastDot != string::npos)
    {
        graphName = graphName.substr(0, lastDot);
    }

    // Create output file
    string outputFileName = "output_" + graphName + "_task_1.txt";
    ofstream outputFile(outputFileName);
    if (!outputFile.is_open())
    {
        cerr << "Error: Could not create output file: " << outputFileName << endl;
        return 1;
    }

    try
    {
        // Create graph from CSV file
        Graph graph(csvFilePath);

        // Validate start vertex
        if (startVertex < 0 || startVertex >= graph.numVertices)
        {
            cerr << "Error: Start vertex " << startVertex << " is out of range [0, "
                 << graph.numVertices - 1 << "]" << endl;
            return 1;
        }

        // Pretty print the graph only for small graphs
        bool isLargeGraph = graph.numVertices > 1000;

        if (!isLargeGraph)
        {
            graph.print();
        }
        else
        {
            cout << "\n============================================" << endl;
            cout << "           GRAPH STRUCTURE                  " << endl;
            cout << "============================================" << endl;
            cout << "Number of Vertices: " << graph.numVertices << endl;
            cout << "Number of Edges:    " << graph.numEdges << endl;
            cout << "============================================" << endl;
            cout << "(Graph too large for detailed printing)" << endl;
        }

        cout << "Graph successfully loaded!" << endl;

        // Run BFS from specified vertex
        cout << "\n============================================" << endl;
        cout << "Starting BFS from vertex " << startVertex << endl;
        cout << "============================================" << endl;

        // ========== CPU BFS ==========
        vector<int> distanceCPU(graph.numVertices);
        vector<bool> visitedCPU(graph.numVertices);

        cout << "\n============================================" << endl;
        cout << "Running CPU BFS from vertex " << startVertex << endl;
        cout << "============================================" << endl;

        auto cpuStart = chrono::steady_clock::now();
        bfsCPU(startVertex, graph, distanceCPU, visitedCPU);
        auto cpuEnd = chrono::steady_clock::now();
        auto cpuDuration = chrono::duration_cast<chrono::microseconds>(cpuEnd - cpuStart).count();

        // Display CPU BFS results
        cout << "\nCPU BFS Results:" << endl;
        cout << "--------------------------------------------" << endl;
        cout << "Execution time: " << cpuDuration / 1000.0 << " ms" << endl;

        int numVisitedCPU = 0;
        int maxDistanceCPU = 0;

        for (int i = 0; i < graph.numVertices; ++i)
        {
            if (distanceCPU[i] != INT_MAX)
            {
                numVisitedCPU++;
                if (distanceCPU[i] > maxDistanceCPU)
                {
                    maxDistanceCPU = distanceCPU[i];
                }
            }
        }

        cout << "Vertices visited: " << numVisitedCPU << " / " << graph.numVertices << endl;
        cout << "Maximum distance: " << maxDistanceCPU << endl;
        cout << "--------------------------------------------" << endl;

        // Create checker with CPU results as reference
        Checker checker(distanceCPU);

        // ========== GPU BFS ==========
        vector<int> distanceGPU(graph.numVertices);
        vector<bool> visitedGPU(graph.numVertices);

        cout << "\n\n============================================" << endl;
        cout << "Running GPU BFS from vertex " << startVertex << endl;
        cout << "============================================" << endl;

        outputFile << "\n\n============================================" << endl;
        outputFile << "Running GPU BFS from vertex " << startVertex << endl;
        outputFile << "============================================" << endl;

        auto gpuStart = chrono::steady_clock::now();
        bool verboseMode = !isLargeGraph;
        bfsGPU(startVertex, graph, distanceGPU, visitedGPU, outputFile, verboseMode);
        auto gpuEnd = chrono::steady_clock::now();
        auto gpuDuration = chrono::duration_cast<chrono::microseconds>(gpuEnd - gpuStart).count();

        // Display GPU BFS results
        cout << "\nGPU BFS Results:" << endl;
        cout << "--------------------------------------------" << endl;
        cout << "Total execution time (including transfers): " << gpuDuration / 1000.0 << " ms" << endl;

        int numVisitedGPU = 0;
        int maxDistanceGPU = 0;

        for (int i = 0; i < graph.numVertices; ++i)
        {
            if (distanceGPU[i] != INT_MAX)
            {
                numVisitedGPU++;
                if (distanceGPU[i] > maxDistanceGPU)
                {
                    maxDistanceGPU = distanceGPU[i];
                }
            }
        }

        cout << "Vertices visited: " << numVisitedGPU << " / " << graph.numVertices << endl;
        cout << "Maximum distance: " << maxDistanceGPU << endl;
        cout << "--------------------------------------------" << endl;

        // ========== Verification ==========
        cout << "\n============================================" << endl;
        cout << "Verification: Comparing CPU and GPU results" << endl;
        cout << "============================================" << endl;

        bool resultsMatch = checker.check(distanceGPU, "GPU BFS");

        // ========== Performance Comparison ==========
        cout << "\n============================================" << endl;
        cout << "         PERFORMANCE COMPARISON             " << endl;
        cout << "============================================" << endl;
        cout << "CPU Execution Time: " << cpuDuration / 1000.0 << " ms" << endl;
        cout << "GPU Execution Time: " << gpuDuration / 1000.0 << " ms" << endl;
        cout << "--------------------------------------------" << endl;

        outputFile << "\n============================================" << endl;
        outputFile << "         PERFORMANCE COMPARISON             " << endl;
        outputFile << "============================================" << endl;
        outputFile << "CPU Execution Time: " << cpuDuration / 1000.0 << " ms" << endl;
        outputFile << "GPU Execution Time: " << gpuDuration / 1000.0 << " ms" << endl;
        outputFile << "--------------------------------------------" << endl;

        if (cpuDuration > gpuDuration)
        {
            double speedup = (double)cpuDuration / gpuDuration;
            cout << "GPU Speedup: " << fixed << setprecision(2) << speedup << "x (GPU is FASTER)" << endl;
            outputFile << "GPU Speedup: " << fixed << setprecision(2) << speedup << "x (GPU is FASTER)" << endl;
        }
        else
        {
            double slowdown = (double)gpuDuration / cpuDuration;
            cout << "GPU Slowdown: " << fixed << setprecision(2) << slowdown << "x (CPU is faster for small graphs)" << endl;
            outputFile << "GPU Slowdown: " << fixed << setprecision(2) << slowdown << "x (CPU is faster for small graphs)" << endl;
        }
        cout << "============================================" << endl;
        outputFile << "============================================" << endl;

        // Show first 20 vertices distances
        cout << "\nDistance from source (first 20 vertices):" << endl;
        cout << "Vertex\tCPU\tGPU" << endl;
        for (int i = 0; i < min(20, graph.numVertices); ++i)
        {
            cout << i << "\t";
            if (distanceCPU[i] == INT_MAX)
                cout << "∞\t";
            else
                cout << distanceCPU[i] << "\t";

            if (distanceGPU[i] == INT_MAX)
                cout << "∞" << endl;
            else
                cout << distanceGPU[i] << endl;
        }
        cout << endl;

        // Close output file
        outputFile.close();
        cout << "\n✓ Output written to file: " << outputFileName << endl;
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        outputFile.close();
        return 1;
    }

    return 0;
}