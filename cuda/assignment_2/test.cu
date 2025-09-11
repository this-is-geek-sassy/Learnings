#include <bits/stdc++.h>
#include <cuda.h>

using namespace std;

int *read_from_csv(const string& filename, vector<int>& data) {
    ifstream file(filename);

    string line;
    int number_of_rows = 0;

    while (getline(file, line))
    {
        string cell;

        for (size_t i = 0; i <= line.size(); i++)
        {
            if (i == line.size() || line[i] == ',') {
                // Trim whitespace from cell
                size_t start = cell.find_first_not_of(" \t\r\n");
                size_t end   = cell.find_last_not_of(" \t\r\n");
                if (start == string::npos) {
                    cell = "";  // cell entirely made up of whitespaces or missing number
                } else {
                    cell = cell.substr(start, end - start + 1);
                }

                // Convert to int and add to flattened data
                if (!cell.empty()) {
                    int value = atoi(cell.c_str());
                    data.push_back(value);
                }
                cell = "";
            } else {
                cell += line[i];
            }
        }
        number_of_rows++;
    }
    file.close();

    // Error check
    if (data.empty()) {
        cout << "Error: No valid data in file: " << filename << endl;
    }

    // Return dimensions
    int *dimensions = (int *)malloc(2 * sizeof(int));
    dimensions[0] = number_of_rows;
    dimensions[1] = 0; // Not needed as per your requirement
    return dimensions;
}

__global__ void dkernel() {
    printf("Hello world\n");
}


int main(int argc, char *argv[]) {

    if (argc != 4 && string(argv[2]) != "-o") {
        cout << "WRONG NUMBER OF ARGUMENTS!!" << endl;
        exit(0);
    }
    string name_of_ip_file = argv[1];
    string name_of_op_file = argv[3];

    vector<int> matrix_alike;

    cout << "You want cpu multiply or gpu multiply?\n\
    Press 0 for CPU, Press 1 for GPU" << endl;
    int c = -99;
    cin >> c;
    while (c!=0 && c!=1) {
        cout << "BAD CHOICE> ENTER AGAIN! >_<" << endl;
        cin >> c;
        // return 0;
    }


    int *dimensions_1 = read_from_csv(name_of_ip_file, matrix_alike);
    int no_of_rows_of_matrix = dimensions_1[0];
    int no_of_cols_of_matrix = dimensions_1[1];

    cout << "no_of_rows_of_matrix: " << no_of_rows_of_matrix << endl;
    cout << "no_of_cols_of_matrix: " << no_of_cols_of_matrix << endl;

    for (size_t i = 0; i < 5; i++)
    {
        dkernel<<<1, 1>>>();
    }
    
    // dkernel<<<1,1>>>();
    cudaDeviceSynchronize();
}