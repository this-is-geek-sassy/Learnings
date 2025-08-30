#include <bits/stdc++.h>
#include <cuda.h>
using namespace std;

int *read_from_csv(const string& filename, vector<int>& data) {
    ifstream file(filename);

    string line;
    vector<vector<double>> __data__;
    int row_size_counter = 0;

    while (getline(file, line))
    {
        vector<double> row;
        
        string cell;

        for (size_t i = 0; i <= line.size(); i++)
        {
            if (i==line.size() || line[i]==',') {
                size_t start = cell.find_first_not_of(" \t\r\n");
                size_t end   = cell.find_last_not_of(" \t\r\n");
                if (start == std::string::npos) {
                    cell = "";  // cell entirely made up of whitespaces or missing number
                } else {
                    cell = cell.substr(start, end - start + 1);
                }

                double value = atof(cell.c_str());
                row.push_back(value);
                cell = "";
            } else {
                cell += line[i];
            }
            // __data__.push_back(row);
        }
        __data__.push_back(row);
        // column_counter++;
        if (row.size() > row_size_counter)
            row_size_counter = row.size();
        
    }
    file.close();

    //running error check
    if (__data__.empty()) {
        cout << "Error: No valid data in file: " << filename << endl;
    }

    int number_of_rows = __data__.size();
    int number_of_cols = __data__[0].size();

    // cout << "number of rows: " << number_of_rows << endl;
    // cout << "number of cols: " << number_of_cols << endl;


    for (size_t i = 0; i < number_of_rows; i++)
    {
        for (size_t j = 0; j < number_of_cols; j++)
        {
            data.push_back(__data__[i][j]);
        }
    }
    int *dimensions = (int *)malloc(2*sizeof(int));
    dimensions[0] = number_of_rows;
    dimensions[1] = number_of_cols;
    return dimensions;
}

void mat_mul (vector<int>& m1, vector<int>& m2, vector<int>& result, int m, int n, int k)
{
    // int m = m1.size(), n = m2.size(), k = m2[0].size();
    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < k; j++)
        {
            for (size_t l = 0; l < n; l++)
            {
                result[i*k + j] += (m1[i*n + l] * m2[l*k + j]);
            }
        }
    }
}

__global__ void gpu_mat_mul(int *d_matrix_1, int *d_matrix_2, int *d_matrix_res, int m, int n, int k) {

    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;

    if (id >= (m*k))
        return;

    // cout << blockIdx.x << " " << threadIdx.x << endl;
    // printf("%d %d\n", blockIdx.x, threadIdx.x);

    unsigned int i = id / k;
    unsigned j = id % k;

    int sum = 0;
    for (size_t l = 0; l < n; l++)
    {
        sum += (d_matrix_1[i*n + l] * d_matrix_2[l*k + j]);
    }
    d_matrix_res[id] = sum;
}

/// TESTING WITH TRANSPOSE
/// BEGIN
vector<int> transpose(const vector<int>& mat, int rows, int cols) {
    vector<int> transposed(cols * rows);

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // element at (i,j) in original goes to (j,i) in transposed
            transposed[j * rows + i] = mat[i * cols + j];
        }
    }

    return transposed;
}
/// END


void write_matrix_to_csv(const string& filename, const vector<int>& mat, int rows, int cols) {
    ofstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << "\n";
        return;
    }

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            file << mat[i * cols + j];
            if (j < cols - 1) 
                file << ",";  // add comma except last element
        }
        file << "\n";
    }

    file.close();
}

int main(int argc, char *argv[]) {

    if (argc != 4) {
        cout << "WRONG NUMBER OF ARGUMENTS!!" << endl;
        exit(0);
    }
    // user only give blocksize, i.e., how many threads should be there per block
    // we will calculae grid_size based on that
    unsigned int block_size = atoi(argv[1]);
    string path_to_mat_a = argv[2];
    string path_to_mat_b = argv[3];

    // int no_of_rows = 5;
    // int no_of_cols = 5;
    // vector<int> matrix_1(no_of_rows*no_of_cols), matrix_2(no_of_rows*no_of_cols);
    vector<int> matrix_1, matrix_2;
    int *d_matrix_1, *d_matrix_2, *d_matrix_result;

    cout << "You want cpu multiply or gpu multiply?\n\
    Press 0 for CPU, Press 1 for GPU" << endl;
    int c = -99;
    cin >> c;
    while (c!=0 && c!=1) {
        cout << "BAD CHOICE> ENTER AGAIN! >_<" << endl;
        cin >> c;
        // return 0;
    }

    // for (size_t i = 0; i < 5; i++)
    // {
    //     // vector<int> v;
    //     for (size_t j = 0; j < 5; j++)
    //     {
    //         // if (i==j)
    //         //     v.push_back(7);
    //         // else
    //         //     v.push_back(0);
    //         // // v.push_back(i+j);
    //         matrix_1[i*no_of_cols + j] = i+j;
    //         matrix_2[i*no_of_cols + j] = i+j;
    //     }
    //     // matrix_1.push_back(v);
    //     // matrix_2.push_back(v);
    // }
    // int * dimensions_1 = read_from_csv("./public_test_cases/matrix1.csv", matrix_1);
    // int *dimensions_2 = read_from_csv("./public_test_cases/matrix2.csv", matrix_2);
    int * dimensions_1 = read_from_csv(path_to_mat_a, matrix_1);
    int *dimensions_2 = read_from_csv(path_to_mat_b, matrix_2);
    
    int no_of_rows_of_matrix_1 = dimensions_1[0];
    int no_of_cols_of_matrix_1 = dimensions_1[1];

    int no_of_rows_of_matrix_2 = dimensions_2[0];
    int no_of_cols_of_matrix_2 = dimensions_2[1];

    // /// TESTING WITH TRANSPOSE
    // /// BEGIN
    // int no_of_rows_of_matrix_2 = dimensions_2[1];
    // int no_of_cols_of_matrix_2 = dimensions_2[0];

    // matrix_2 = transpose(matrix_2, no_of_rows_of_matrix_2, no_of_cols_of_matrix_2);
    // /// END

    bool dims_match = (no_of_cols_of_matrix_1 == no_of_rows_of_matrix_2);
    bool can_transpose_1 = (no_of_rows_of_matrix_1 == no_of_rows_of_matrix_2);
    bool can_transpose_2 = (no_of_cols_of_matrix_1 == no_of_cols_of_matrix_2);

    if (!dims_match) {
        if (!can_transpose_1 && !can_transpose_2) {
            cout << "DIMENSION MISMATCH, CANNOT PROCEED!!" << endl;
            return -1;
        }

        // Calculate scalar multiplications for both options
        long long cost1 = LLONG_MAX, cost2 = LLONG_MAX;
        if (can_transpose_1) {
            // If we transpose matrix 1
            cost1 = (long long)no_of_cols_of_matrix_1 * no_of_rows_of_matrix_2 * no_of_cols_of_matrix_2;
        }
        if (can_transpose_2) {
            // If we transpose matrix 2
            cost2 = (long long)no_of_rows_of_matrix_1 * no_of_cols_of_matrix_1 * no_of_rows_of_matrix_2;
        }

        cout << "Dimension mismatch! But multiplication is possible if one matrix is transposed." << endl;
        if (can_transpose_1 && can_transpose_2) {
            cout << "Transpose matrix 1 (A) or matrix 2 (B)?" << endl;
            cout << "1: Transpose A (cost: " << cost1 << ")" << endl;
            cout << "2: Transpose B (cost: " << cost2 << ")" << endl;
            cout << "0: Abort" << endl;
            int choice = 0;
            do {
                cout << "Enter your choice: ";
                cin >> choice;
            } while (choice != 0 && choice != 1 && choice != 2);
            if (choice == 0) return -1;
            if ((choice == 1 && cost1 <= cost2) || (choice == 2 && cost2 < cost1)) {
                if (choice == 1) {
                    matrix_1 = transpose(matrix_1, no_of_rows_of_matrix_1, no_of_cols_of_matrix_1);
                    swap(no_of_rows_of_matrix_1, no_of_cols_of_matrix_1);
                } else {
                    matrix_2 = transpose(matrix_2, no_of_rows_of_matrix_2, no_of_cols_of_matrix_2);
                    swap(no_of_rows_of_matrix_2, no_of_cols_of_matrix_2);
                }
            }
        } else if (can_transpose_1) {
            cout << "Transpose matrix 1 (A) to proceed? (y/n): ";
            char ans; cin >> ans;
            if (ans == 'y' || ans == 'Y') {
                matrix_1 = transpose(matrix_1, no_of_rows_of_matrix_1, no_of_cols_of_matrix_1);
                swap(no_of_rows_of_matrix_1, no_of_cols_of_matrix_1);
            } else {
                cout << "Aborted by user." << endl;
                return -1;
            }
        } else if (can_transpose_2) {
            cout << "Transpose matrix 2 (B) to proceed? (y/n): ";
            char ans; cin >> ans;
            if (ans == 'y' || ans == 'Y') {
                matrix_2 = transpose(matrix_2, no_of_rows_of_matrix_2, no_of_cols_of_matrix_2);
                swap(no_of_rows_of_matrix_2, no_of_cols_of_matrix_2);
            } else {
                cout << "Aborted by user." << endl;
                return -1;
            }
        }
    }

    vector<int> result(no_of_rows_of_matrix_1 * no_of_cols_of_matrix_2);
    // result.resize(no_of_rows, vector<int>(no_of_cols, 0));

    if (c == 0) {
        auto start = chrono::high_resolution_clock::now();
        mat_mul(matrix_1, matrix_2, result, no_of_rows_of_matrix_1, no_of_cols_of_matrix_1, no_of_cols_of_matrix_2);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        cout << "CPU function took " << duration.count()*1000 << " micro seconds\n";
    }

    // // printing
    // for (size_t i = 0; i < no_of_rows_of_matrix_1; i++)
    // {
    //     for (size_t j = 0; j < no_of_cols_of_matrix_1; j++)
    //     {
    //         cout << matrix_1[i*no_of_cols_of_matrix_1 + j] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;
    // for (size_t i = 0; i < no_of_rows_of_matrix_2; i++)
    // {
    //     for (size_t j = 0; j < no_of_cols_of_matrix_2; j++)
    //     {
    //         cout << matrix_2[i*no_of_cols_of_matrix_2 + j] << " ";
    //     }
    //     cout << endl;
    // }
    // cout << endl;

    // return 0;

    else if (c == 1) {
        // allocation on GPu
        cudaMalloc(&d_matrix_1, no_of_rows_of_matrix_1*no_of_cols_of_matrix_1*sizeof(int));
        cudaMalloc(&d_matrix_2, no_of_rows_of_matrix_2*no_of_cols_of_matrix_2*sizeof(int));
        cudaMalloc(&d_matrix_result, no_of_rows_of_matrix_1*no_of_cols_of_matrix_2*sizeof(int));

        //transferring data to GPU
        cudaMemcpy(d_matrix_1, matrix_1.data(), no_of_rows_of_matrix_1*no_of_cols_of_matrix_1*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix_2, matrix_2.data(), no_of_rows_of_matrix_2*no_of_cols_of_matrix_2*sizeof(int), cudaMemcpyHostToDevice);

        // setting launch config: 
        //unsigned int block_size : already available
        unsigned int grid_size = ceil((1.0*no_of_rows_of_matrix_1*no_of_cols_of_matrix_2)/block_size);

        //timing
        // auto start = chrono::high_resolution_clock::now();
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        gpu_mat_mul<<<grid_size, block_size>>> (d_matrix_1, d_matrix_2, d_matrix_result, no_of_rows_of_matrix_1, no_of_cols_of_matrix_1, no_of_cols_of_matrix_2);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        printf("Kernel elapsed time: %.3f microseconds\n", ms * 1000);

        // cudaDeviceSynchronize();
        // auto end = chrono::high_resolution_clock::now();
        // chrono::duration<double, milli> duration = end-start;

        // cout << "GPU function took " << duration.count() << " ms\n";

        // transferring resultant matrix into cpu
        cudaMemcpy(result.data(), d_matrix_result, no_of_rows_of_matrix_1 * no_of_cols_of_matrix_2 * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // cout << *result.data() << endl;

    // for (size_t i = 0; i < no_of_rows_of_matrix_1; i++)
    // {
    //     for (size_t j = 0; j < no_of_cols_of_matrix_2; j++)
    //     {
    //         cout << result[i*no_of_cols_of_matrix_2 + j] << " ";
    //     }
    //     cout << endl;
    // }
    write_matrix_to_csv("./results/result_1_1.csv", result, no_of_rows_of_matrix_1, no_of_cols_of_matrix_2);
    
    return 0;
}