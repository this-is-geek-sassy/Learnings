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

    // cout << blockIdx.x << " " << threadIdx.x << endl;
    // printf("%d %d\n", blockIdx.x, threadIdx.x);

    unsigned int i = blockIdx.x;
    unsigned j = threadIdx.x;

    for (size_t l = 0; l < n; l++)
    {
        d_matrix_res[id] += (d_matrix_1[i*n + l] * d_matrix_2[l*k + j]);
    }
    
}

/// TESTING WITH TRANSPOSE
/// BEGIN
std::vector<int> transpose(const std::vector<int>& mat, int rows, int cols) {
    std::vector<int> transposed(cols * rows);

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

int main() {
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
    int * dimensions_1 = read_from_csv("./public_test_cases/matrix_a.csv", matrix_1);
    int *dimensions_2 = read_from_csv("./public_test_cases/matrix_b.csv", matrix_2);
    
    int no_of_rows_of_matrix_1 = dimensions_1[0];
    int no_of_cols_of_matrix_1 = dimensions_1[1];

    // int no_of_rows_of_matrix_2 = dimensions_2[0];
    // int no_of_cols_of_matrix_2 = dimensions_2[1];

    /// TESTING WITH TRANSPOSE
    /// BEGIN
    int no_of_rows_of_matrix_2 = dimensions_2[1];
    int no_of_cols_of_matrix_2 = dimensions_2[0];

    matrix_2 = transpose(matrix_2, no_of_rows_of_matrix_2, no_of_cols_of_matrix_2);
    /// END

    if (no_of_cols_of_matrix_1 != no_of_rows_of_matrix_2) {
        cout << "DIMENSION MISMATCH, CANNOT PROCEED!!" << endl;
        return -1;
    }

    vector<int> result(no_of_rows_of_matrix_1 * no_of_cols_of_matrix_2);
    // result.resize(no_of_rows, vector<int>(no_of_cols, 0));

    if (c == 0) {
        auto start = chrono::high_resolution_clock::now();
        mat_mul(matrix_1, matrix_2, result, no_of_rows_of_matrix_1, no_of_cols_of_matrix_1, no_of_cols_of_matrix_2);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        cout << "CPU function took " << duration.count() << " ms\n";
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

        //timing
        auto start = chrono::high_resolution_clock::now();
        gpu_mat_mul<<<no_of_rows_of_matrix_1,no_of_cols_of_matrix_2>>> (d_matrix_1, d_matrix_2, d_matrix_result, no_of_rows_of_matrix_1, no_of_cols_of_matrix_1, no_of_cols_of_matrix_2);
        cudaDeviceSynchronize();
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end-start;

        cout << "GPU function took " << duration.count() << " ms\n";

        // transferring resultant matrix into cpu
        cudaMemcpy(result.data(), d_matrix_result, no_of_rows_of_matrix_1 * no_of_cols_of_matrix_2 * sizeof(int), cudaMemcpyDeviceToHost);
    }

    // cout << *result.data() << endl;

    for (size_t i = 0; i < no_of_rows_of_matrix_1; i++)
    {
        for (size_t j = 0; j < no_of_cols_of_matrix_2; j++)
        {
            cout << result[i*no_of_cols_of_matrix_2 + j] << " ";
        }
        cout << endl;
    }
    write_matrix_to_csv("./result.csv", result, no_of_rows_of_matrix_1, no_of_cols_of_matrix_2);
    
    return 0;
}