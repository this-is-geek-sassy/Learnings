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

// GPU Kernel: Process current frontier with compact neighbor data
__global__ void bfsKernel_Compact(int *compactNeighbors, int *compactOffsets, int *compactSizes,
                                  int *distance, int queueSize, int *currentQueue,
                                  int *nextQueueSize, int *nextQueue, int level)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < queueSize)
    {
        // Use compact offset: index into compactNeighbors specific to this frontier
        int offset = compactOffsets[tid];
        int numNeighbors = compactSizes[tid];

        // Explore neighbors from compact array
        for (int i = 0; i < numNeighbors; ++i)
        {
            int neighbor = compactNeighbors[offset + i];

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
    // If features are not loaded, use node ID difference as fallback
    if (G.features.empty() || G.featureDimensions == 0)
    {
        return abs(node - start);
    }

    // Calculate actual Euclidean distance using feature vectors
    double sum = 0.0;
    int dims = G.featureDimensions;

    for (int d = 0; d < dims; ++d)
    {
        float diff = G.features[node * dims + d] - G.features[start * dims + d];
        sum += diff * diff;
    }

    return sqrt(sum);
}

// CUDA Graph-based GPU BFS with stream capture
void bfsGPU_Graph(int start, Graph &G, vector<int> &distance, vector<bool> &visited,
                  ofstream &outputFile, bool verbose = true)
{
    const int n_blocks = (G.numVertices + N_THREADS_PER_BLOCK - 1) / N_THREADS_PER_BLOCK;

    // Device pointers - NO d_adjacencyList!
    int *d_edgesOffset;
    int *d_edgesSize;
    int *d_firstQueue;
    int *d_secondQueue;
    int *d_nextQueueSize;
    int *d_distance;

    // Compact neighbor data for current frontier only
    int *d_compactNeighbors;
    int *d_compactOffsets;
    int *d_compactSizes;

    // Pinned host memory for async transfers
    int *h_queueSize;
    int *h_nextQueueSize;

    // Pinned host memory for compact neighbor data (pre-allocated for max size)
    int *h_compactNeighbors;
    int *h_compactOffsets;
    int *h_compactSizes;

    // Host variables
    int currentQueueSize = 1;
    const int NEXT_QUEUE_SIZE = 0;
    int level = 0;

    // For tracking nodes at each level
    vector<vector<int>> nodesAtLevel;

    // Allocate device memory (NO adjacency list!)
    const int size = G.numVertices * sizeof(int);

    cudaMalloc((void **)&d_edgesOffset, size);
    cudaMalloc((void **)&d_edgesSize, size);
    cudaMalloc((void **)&d_firstQueue, size);
    cudaMalloc((void **)&d_secondQueue, size);
    cudaMalloc((void **)&d_distance, size);
    cudaMalloc((void **)&d_nextQueueSize, sizeof(int));

    // Allocate space for compact neighbor data (max possible per iteration)
    cudaMalloc((void **)&d_compactNeighbors, G.numEdges * sizeof(int));
    cudaMalloc((void **)&d_compactOffsets, size);
    cudaMalloc((void **)&d_compactSizes, size);

    // Allocate pinned host memory for async operations
    cudaMallocHost((void **)&h_queueSize, sizeof(int));
    cudaMallocHost((void **)&h_nextQueueSize, sizeof(int));

    // Allocate pinned memory for compact data (max size to avoid reallocation)
    cudaMallocHost((void **)&h_compactNeighbors, G.numEdges * sizeof(int));
    cudaMallocHost((void **)&h_compactOffsets, G.numVertices * sizeof(int));
    cudaMallocHost((void **)&h_compactSizes, G.numVertices * sizeof(int));

    // Transfer graph METADATA only (not adjacency list!)
    if (verbose)
        cout << "\n[Phase 1] CPU -> GPU: Transferring graph metadata (NO adjacency list transfer)..." << endl;
    auto transferStart = chrono::steady_clock::now();

    // Use synchronous copy for initial setup (before stream capture)
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

    // BFS traversal loop with CUDA Graph stream capture and on-demand neighbor transfer
    if (verbose)
    {
        cout << "\n[Phase 3] Starting BFS with CUDA Graph + on-demand neighbor transfer (pinned memory)..." << endl;
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

    // Create streams and CUDA Graph
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t graphExec = nullptr;
    cudaGraphNode_t kernelNode;
    bool graphCaptured = false;

    while (currentQueueSize > 0)
    {
        // Determine which queue to use (ping-pong between two queues)
        int *d_currentQueue = (level % 2 == 0) ? d_firstQueue : d_secondQueue;
        int *d_nextQueue = (level % 2 == 0) ? d_secondQueue : d_firstQueue;

        // Copy current queue to host to track nodes at this level
        vector<int> currentLevelNodes(currentQueueSize);
        cudaMemcpy(currentLevelNodes.data(), d_currentQueue, currentQueueSize * sizeof(int), cudaMemcpyDeviceToHost);
        nodesAtLevel.push_back(currentLevelNodes);

        // Pack neighbors for ONLY the nodes in current frontier into pinned memory
        int currentOffset = 0;
        int totalNeighbors = 0;

        for (int i = 0; i < currentQueueSize; ++i)
        {
            int node = currentLevelNodes[i];
            int offset = G.edgesOffset[node];
            int numNeighbors = G.edgesSize[node];

            h_compactOffsets[i] = currentOffset;
            h_compactSizes[i] = numNeighbors;

            // Copy this node's neighbors directly into pinned array
            memcpy(&h_compactNeighbors[currentOffset], &G.adjacencyList[offset], numNeighbors * sizeof(int));

            currentOffset += numNeighbors;
        }
        totalNeighbors = currentOffset;

        // For first iteration, capture the pattern with stream capture
        if (!graphCaptured)
        {
            if (verbose)
                cout << "[CUDA Graph] Capturing BFS iteration pattern with pinned memory transfers..." << endl;

            // Begin stream capture
            cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);

            // Capture transfer of compact neighbor data (using pinned memory)
            cudaMemcpyAsync(d_compactNeighbors, h_compactNeighbors, totalNeighbors * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_compactOffsets, h_compactOffsets, currentQueueSize * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_compactSizes, h_compactSizes, currentQueueSize * sizeof(int),
                            cudaMemcpyHostToDevice, stream);

            // Capture kernel launch with compact data
            bfsKernel_Compact<<<n_blocks, N_THREADS_PER_BLOCK, 0, stream>>>(
                d_compactNeighbors, d_compactOffsets, d_compactSizes, d_distance,
                currentQueueSize, d_currentQueue, d_nextQueueSize, d_nextQueue, level);

            // Capture async memcpy of next queue size
            cudaMemcpyAsync(h_nextQueueSize, d_nextQueueSize, sizeof(int),
                            cudaMemcpyDeviceToHost, stream);

            // End stream capture
            cudaStreamEndCapture(stream, &graph);

            // Instantiate the captured graph
            cudaGraphInstantiate(&graphExec, graph, nullptr, nullptr, 0);

            // Get the kernel node for later updates
            size_t numNodes = 0;
            cudaGraphGetNodes(graph, nullptr, &numNodes);
            cudaGraphNode_t *nodes = new cudaGraphNode_t[numNodes];
            cudaGraphGetNodes(graph, nodes, &numNodes);

            // Find the kernel node
            for (size_t i = 0; i < numNodes; i++)
            {
                cudaGraphNodeType type;
                cudaGraphNodeGetType(nodes[i], &type);
                if (type == cudaGraphNodeTypeKernel)
                {
                    kernelNode = nodes[i];
                    break;
                }
            }
            delete[] nodes;

            graphCaptured = true;
            if (verbose)
                cout << "[CUDA Graph] Graph instantiated successfully! Will reuse for subsequent iterations." << endl;
        }
        else
        {
            // Transfer updated compact data for this iteration (async, outside graph)
            // This is unavoidable because frontier content changes each iteration
            cudaMemcpyAsync(d_compactNeighbors, h_compactNeighbors, totalNeighbors * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_compactOffsets, h_compactOffsets, currentQueueSize * sizeof(int),
                            cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(d_compactSizes, h_compactSizes, currentQueueSize * sizeof(int),
                            cudaMemcpyHostToDevice, stream);

            // Wait for transfers to complete before launching graph
            cudaStreamSynchronize(stream);

            // Update kernel parameters for subsequent iterations
            cudaKernelNodeParams kernelParams;
            cudaGraphKernelNodeGetParams(kernelNode, &kernelParams);

            // Update the parameters
            void *kernelArgs[] = {&d_compactNeighbors, &d_compactOffsets, &d_compactSizes, &d_distance,
                                  &currentQueueSize, &d_currentQueue, &d_nextQueueSize, &d_nextQueue, &level};
            kernelParams.kernelParams = kernelArgs;

            cudaGraphExecKernelNodeSetParams(graphExec, kernelNode, &kernelParams);
        }

        // Launch the CUDA graph (replays captured pattern)
        cudaGraphLaunch(graphExec, stream);
        cudaStreamSynchronize(stream);

        // Get next queue size from pinned memory
        currentQueueSize = *h_nextQueueSize;

        // Reset next queue size for next iteration
        *h_nextQueueSize = NEXT_QUEUE_SIZE;
        cudaMemcpy(d_nextQueueSize, h_nextQueueSize, sizeof(int), cudaMemcpyHostToDevice); // Find closest and farthest nodes at this level based on Euclidean distance
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
        snprintf(buffer, sizeof(buffer),
                 "| %-5d | %-21d | Farthest node = %d, Dist=%.1f\n| %5s | %21s | Closest node = %d, Dist=%.1f",
                 level, currentQueueSize, farthestNode, farthestDist, "", "", closestNode, closestDist);

        string output(buffer);
        cout << output << endl;
        cout << separator << endl;
        outputFile << output << endl;
        outputFile << separator << endl;

        // Update level counter
        ++level;

        // Note: Queue pointers switch automatically via (level % 2) logic
        // Graph will be re-executed with updated queue data
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
    if (graphExec)
        cudaGraphExecDestroy(graphExec);
    if (graphCaptured)
        cudaGraphDestroy(graph);

    cudaStreamDestroy(stream);

    cudaFree(d_edgesOffset);
    cudaFree(d_edgesSize);
    cudaFree(d_firstQueue);
    cudaFree(d_secondQueue);
    cudaFree(d_distance);
    cudaFree(d_nextQueueSize);
    cudaFree(d_compactNeighbors);
    cudaFree(d_compactOffsets);
    cudaFree(d_compactSizes);

    cudaFreeHost(h_queueSize);
    cudaFreeHost(h_nextQueueSize);
    cudaFreeHost(h_compactNeighbors);
    cudaFreeHost(h_compactOffsets);
    cudaFreeHost(h_compactSizes);

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
        cerr << "Usage: " << argv[0] << " <path_to_csv_file> [start_vertex] [path_to_fvecs_file]" << endl;
        cerr << "Example: " << argv[0] << " ../graph_100_8.csv" << endl;
        cerr << "Example: " << argv[0] << " ../graph_10000_16.csv 3263" << endl;
        cerr << "Example: " << argv[0] << " ../graph_10000_16.csv 3263 ../sift/sift_base.fvecs" << endl;
        return 1;
    }

    string csvFilePath = argv[1];
    int startVertex = 0;       // Default start vertex
    string fvecsFilePath = ""; // Optional feature vectors file

    // Parse optional start vertex argument
    if (argc >= 3)
    {
        startVertex = atoi(argv[2]);
    }

    // Parse optional .fvecs file path
    if (argc >= 4)
    {
        fvecsFilePath = argv[3];
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
    string outputFileName = "output_" + graphName + "_graph.txt";
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

        // Load feature vectors if provided
        if (!fvecsFilePath.empty())
        {
            graph.loadFeatures(fvecsFilePath);
            cout << "Using actual Euclidean distances from feature vectors" << endl;
        }
        else
        {
            cout << "No feature vectors provided. Using node ID difference as distance metric." << endl;
            graph.featureDimensions = 0; // Mark as not loaded
        }

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
        cout << "Running GPU BFS (CUDA Graph) from vertex " << startVertex << endl;
        cout << "============================================" << endl;

        outputFile << "\n\n============================================" << endl;
        outputFile << "Running GPU BFS (CUDA Graph) from vertex " << startVertex << endl;
        outputFile << "============================================" << endl;

        auto gpuStart = chrono::steady_clock::now();
        bool verboseMode = !isLargeGraph;
        bfsGPU_Graph(startVertex, graph, distanceGPU, visitedGPU, outputFile, verboseMode);
        auto gpuEnd = chrono::steady_clock::now();
        auto gpuDuration = chrono::duration_cast<chrono::microseconds>(gpuEnd - gpuStart).count();

        // Display GPU BFS results
        cout << "\nGPU BFS Results (CUDA Graph):" << endl;
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

        bool resultsMatch = checker.check(distanceGPU, "GPU BFS (CUDA Graph)");

        // ========== Performance Comparison ==========
        cout << "\n============================================" << endl;
        cout << "         PERFORMANCE COMPARISON             " << endl;
        cout << "============================================" << endl;
        cout << "CPU Execution Time: " << cpuDuration / 1000.0 << " ms" << endl;
        cout << "GPU Execution Time (CUDA Graph): " << gpuDuration / 1000.0 << " ms" << endl;
        cout << "--------------------------------------------" << endl;

        outputFile << "\n============================================" << endl;
        outputFile << "         PERFORMANCE COMPARISON             " << endl;
        outputFile << "============================================" << endl;
        outputFile << "CPU Execution Time: " << cpuDuration / 1000.0 << " ms" << endl;
        outputFile << "GPU Execution Time (CUDA Graph): " << gpuDuration / 1000.0 << " ms" << endl;
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