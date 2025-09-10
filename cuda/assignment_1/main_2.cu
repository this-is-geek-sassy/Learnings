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
                if (start == string::npos) {
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

// CPU matrix multiplication without tilling
void ordinary_mat_mul (vector<int>& m1, vector<int>& m2, vector<int>& result, int m, int n, int k, int tile_D)
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

// CPU tile_mat_mul
void tile_mat_mul (vector<int>& m1, vector<int>& m2, vector<int>& result, int m, int n, int k, int tile_D)
{
    // int m = m1.size(), n = m2.size(), k = m2[0].size();
    for (size_t ii = 0; ii < m; ii += tile_D)
    {
        for (size_t jj = 0; jj < k; jj += tile_D)
        {
            for (size_t ll = 0; ll < n; ll += tile_D)
            {
                // INNER LOOPS STARTING
                for (size_t i=ii; i<ii+tile_D; i++) {

                    if (i >= m)
                        continue;

                    for(size_t j=jj; j<jj+tile_D; j++) {

                        if (j >= k)
                            continue;
                        
                        // int sum = 0;
                        for(size_t l=ll; l<ll+tile_D; l++) {

                            if (l >= n)
                                continue;

                            result[i*k + j] += (m1[i*n + l] * m2[l*k + j]);
                        }
                        // result[i*k + j] = sum;
                    }
                }
                // return;
            }
        }
    }
}

// Tile matrix multiple multiplication
__global__ void gpu_tile_mat_mul(int *d_matrix_1, int *d_matrix_2, int *d_matrix_res, int m, int n, int k, int tile_D) {

    extern __shared__ int p_tile[];

    int *v_tile_1 = p_tile;
    int *v_tile_2 = p_tile + tile_D*tile_D;

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    // if (i >= m)
    //     return;
    // if (j >= k)
    //     return;
    
    int sum = 0;
    for (size_t t = 0; t < ceil((double)n/tile_D); t++) {

        // Load A tile
        if (i < m && (t*tile_D + threadIdx.y) < n) {
            v_tile_1[threadIdx.x*tile_D + threadIdx.y] =
                d_matrix_1[i*n + (t*tile_D + threadIdx.y)];
        } else {
            v_tile_1[threadIdx.x*tile_D + threadIdx.y] = 0;
        }

        // Load B tile
        if (j < k && (t*tile_D + threadIdx.x) < n) {
            v_tile_2[threadIdx.x*tile_D + threadIdx.y] =
                d_matrix_2[(t*tile_D + threadIdx.x)*k + j];
        } else {
            v_tile_2[threadIdx.x*tile_D + threadIdx.y] = 0;
        }

        __syncthreads(); // now safe to use v_tile_1, v_tile_2 for this tile

        // multiply-accumulate with this tile
        // int sum = 0;
        for (int l = 0; l < tile_D; l++) {
            sum += v_tile_1[threadIdx.x * tile_D + l] * v_tile_2[l * tile_D + threadIdx.y];
        }

        __syncthreads(); // before loading next tile
    }
    if (i < m && j < k) {
        d_matrix_res[i*k + j] = sum;
    }

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

    int tile_m = atoi(argv[1]);
    int tile_n = tile_m;   // taking square tiles for now
    string path_to_mat_a = argv[2];
    string path_to_mat_b = argv[3];

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
        ordinary_mat_mul(matrix_1, matrix_2, result, no_of_rows_of_matrix_1, no_of_cols_of_matrix_1, no_of_cols_of_matrix_2, tile_m);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        cout << "CPU function took " << duration.count()*1000 << " micro seconds\n";
    }
    else if (c==1) {
        // mem allocation on gpu
        cudaMalloc(&d_matrix_1, no_of_rows_of_matrix_1*no_of_cols_of_matrix_1*sizeof(int));
        cudaMalloc(&d_matrix_2, no_of_rows_of_matrix_2*no_of_cols_of_matrix_2*sizeof(int));
        cudaMalloc(&d_matrix_result, no_of_rows_of_matrix_1*no_of_cols_of_matrix_2*sizeof(int));

        //transferring data to GPU
        cudaMemcpy(d_matrix_1, matrix_1.data(), no_of_rows_of_matrix_1*no_of_cols_of_matrix_1*sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_matrix_2, matrix_2.data(), no_of_rows_of_matrix_2*no_of_cols_of_matrix_2*sizeof(int), cudaMemcpyHostToDevice);

        // setting launch config:
        unsigned int no_of_blocks_x = ceil((1.0*no_of_rows_of_matrix_1) / tile_m);
        unsigned int no_of_blocks_y = ceil((1.0*no_of_cols_of_matrix_1) / tile_n);
        unsigned int shared_memory_size = tile_m * tile_n * sizeof(int) * 2;

        cout << "no of blocks_x: " << no_of_blocks_x << endl;
        cout << "no of blocks_y: " << no_of_blocks_y << endl;
        cout << "shared mem size: " << shared_memory_size << endl;
        
        // block & grid creation:
        dim3 block(tile_m, tile_n);
        dim3 grid(no_of_blocks_x, no_of_blocks_y);

        // timing
        // auto start = chrono::high_resolution_clock::now();
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        gpu_tile_mat_mul<<<grid, block, shared_memory_size>>>(d_matrix_1, d_matrix_2, d_matrix_result, no_of_rows_of_matrix_1, no_of_cols_of_matrix_1, no_of_cols_of_matrix_2, tile_m);

        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        printf("Kernel elapsed time: %.3f microseconds\n", ms * 1000);
        // Write kernel time to output file
        ofstream timefile("./public_test_cases/output_2_CS24MTECH12001.txt");
        if (timefile.is_open()) {
            timefile << "Kernel elapsed time: " << fixed << setprecision(3) << (ms * 1000) << " microseconds\n";
            timefile.close();
        } else {
            cerr << "Could not open output_2_CS24MTECH12001.txt for writing kernel time!\n";
        }

        // cudaDeviceSynchronize();
        // auto end = chrono::high_resolution_clock::now();
        // chrono::duration<double, milli> duration = end-start;

        // cout << "GPU function took " << duration.count() << " ms\n";

        // transferring resultant matrix into cpu
        cudaMemcpy(result.data(), d_matrix_result, no_of_rows_of_matrix_1 * no_of_cols_of_matrix_2 * sizeof(int), cudaMemcpyDeviceToHost);

        // Free GPU memory
        cudaFree(d_matrix_1);
        cudaFree(d_matrix_2);
        cudaFree(d_matrix_result);
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
    // write_matrix_to_csv("./results/output_2_CS24MTECH12001.csv", result, no_of_rows_of_matrix_1, no_of_cols_of_matrix_2);
    write_matrix_to_csv("./public_test_cases/output_2_CS24MTECH12001.csv", result, no_of_rows_of_matrix_1, no_of_cols_of_matrix_2);
    
    return 0;
}