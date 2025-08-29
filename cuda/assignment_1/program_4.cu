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

// CPU tile transpose
vector<int> tile_transpose(const vector<int>& mat, int rows, int cols, int tile_D1, int tile_D2) {
    vector<int> transposed(cols * rows);

    for (int ii = 0; ii < rows; ii += tile_D1) {
        for (int jj = 0; jj < cols; jj += tile_D2) {

            for (int i = ii; i < ii+ tile_D1; i++) {
                if (i >= rows) 
                    continue;

                for (int j = jj; j < jj + tile_D2; j++) {
                    if (j >= cols) 
                        continue;

                    // element at (i,j) in original goes to (j,i) in transposed
                    transposed[j * rows + i] = mat[i * cols + j];
                }
            }
        }
    }

    return transposed;
}


// GPU tile transpose (to be)
__global__ void gpu_tile_transpose(int *d_matrix_1, int *d_matrix_res, int rows, int cols, int tile_D1, int tile_D2) {

    extern __shared__ int tile[];

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;

    if (i >= rows)
        return;
    if (j >= cols)
        return;

    tile[threadIdx.y*tile_D1 + threadIdx.x] = d_matrix_1[i*cols + j];
    __syncthreads();

    // d_matrix_res[j*rows + i] = d_matrix_1[i*cols + j];

    // unsigned int ti = blockIdx.y*blockDim.y + threadIdx.y;
    // unsigned int tj = blockIdx.x*blockDim.x + threadIdx.x;
    // if (ti >= cols)
    //     return;
    // if (tj >= rows)
    //     return;
    
    // d_matrix_res[tj*rows + ti] = tile[threadIdx.y*tile_D1 + threadIdx.x];

    // DEBUG: Print the transposed tile (only thread (0,0) of block (0,0) does this)
    if (threadIdx.x == 0 && threadIdx.y == 0 && blockIdx.x == 0 && blockIdx.y == 0) {
        printf("\n=== Block (%d, %d) - Transposed Tile in Shared Memory ===\n", blockIdx.x, blockIdx.y);
        printf("Original block covers rows [%d-%d], cols [%d-%d]\n", 
               blockIdx.x * tile_D1, min((int)rows-1, (int)(blockIdx.x * tile_D1 + tile_D1 - 1)),
               blockIdx.y * tile_D2, min((int)cols-1, (int)(blockIdx.y * tile_D2 + tile_D2 - 1)));
        
        // Print the tile row by row (tile is now tile_D2 x tile_D1 due to transpose)
        for (int i = 0; i < tile_D2 && (blockIdx.y * tile_D2 + i) < cols; i++) {
            printf("Row %d: ", i);
            for (int j = 0; j < tile_D1 && (blockIdx.x * tile_D1 + j) < rows; j++) {
                printf("%4d ", tile[i * tile_D1 + j]);
            }
            printf("\n");
        }
        printf("=====================================\n");
    }
    
    __syncthreads(); // Make sure printing is done before proceeding
}


int main(int argc, char *argv[]) {

    if (argc != 3) {
        cout << "WRONG NUMBER OF ARGUMENTS!!" << endl;
        exit(0);
    }

    int tile_m = atoi(argv[1]);
    int tile_n = atoi(argv[2]);

    // cout << argc << endl;
    // cout << tile_m << " " << tile_n << endl;

    vector<int> matrix_1;
    int *d_matrix_1, *d_matrix_res;

    cout << "You want to transpose on CPU or GPU\n\
    Press 0 for CPU, Press 1 for GPU" << endl;
    int c = -99;
    cin >> c;
    while (c!=0 && c!=1) {
        cout << "BAD CHOICE> ENTER AGAIN! >_<" << endl;
        cin >> c;
        // return 0;
    }
    int * dimensions_1 = read_from_csv("./public_test_cases/matrix1.csv", matrix_1);
    int no_of_rows_of_matrix_1 = dimensions_1[0];
    int no_of_cols_of_matrix_1 = dimensions_1[1];

    vector<int> matrix_res(no_of_cols_of_matrix_1 * no_of_rows_of_matrix_1);

    if (c==0) {
        auto start = chrono::high_resolution_clock::now();
        matrix_res = tile_transpose(matrix_1, no_of_rows_of_matrix_1, no_of_cols_of_matrix_1, tile_m, tile_n);
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end - start;
        cout << "CPU function took " << duration.count() << " ms\n";
    }
    else if (c==1) {
        // memory allocation on GPU
        cudaMalloc(&d_matrix_1, no_of_rows_of_matrix_1*no_of_cols_of_matrix_1*sizeof(int));
        cudaMalloc(&d_matrix_res, no_of_rows_of_matrix_1*no_of_cols_of_matrix_1*sizeof(int));

        // transfering matcrix_1 to GPU
        cudaMemcpy(d_matrix_1, matrix_1.data(), no_of_rows_of_matrix_1*no_of_cols_of_matrix_1*sizeof(int), cudaMemcpyHostToDevice);

        unsigned int no_of_blocks_x = ceil((1.0*no_of_rows_of_matrix_1) / tile_m);
        unsigned int no_of_blocks_y = ceil((1.0*no_of_cols_of_matrix_1) / tile_n);

        unsigned int shared_memory_size = tile_m * tile_n * sizeof(int);

        cout << "no of blocks_x: " << no_of_blocks_x << endl;
        cout << "no of blocks_y: " << no_of_blocks_y << endl;
        cout << "shared mem size: " << shared_memory_size << endl;

        // block & grid creation
        dim3 block(tile_m, tile_n);
        dim3 grid(no_of_blocks_x, no_of_blocks_y);

        //timing
        auto start = chrono::high_resolution_clock::now();

        gpu_tile_transpose<<<grid, block, shared_memory_size>>>(d_matrix_1, d_matrix_res, no_of_rows_of_matrix_1, no_of_cols_of_matrix_1, tile_m, tile_n);
        cudaDeviceSynchronize();

        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double, milli> duration = end-start;

        cout << "GPU function took " << duration.count() << " ms\n";

        //transferring resultant matrix into host
        cudaMemcpy(matrix_res.data(), d_matrix_res, no_of_rows_of_matrix_1 * no_of_cols_of_matrix_1 * sizeof(int), cudaMemcpyDeviceToHost);
    }
    // printing the op
    for (size_t i = 0; i < no_of_cols_of_matrix_1; i++)
    {
        for (size_t j = 0; j < no_of_rows_of_matrix_1; j++)
        {
            cout << matrix_res[i*no_of_rows_of_matrix_1 + j] << " ";
        }
        cout << endl;
    }
    write_matrix_to_csv("./result.csv", matrix_res, no_of_cols_of_matrix_1, no_of_rows_of_matrix_1);
    
    return 0;
}