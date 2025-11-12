#include <iostream>
#include <stdexcept>
#include "graph.h"

using namespace std;

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
    }
    catch (const exception &e)
    {
        cerr << "Error: " << e.what() << endl;
        return 1;
    }

    return 0;
}