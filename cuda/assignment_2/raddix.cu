#include <bits/stdc++.h>
#include <cuda.h>

using namespace std;

int *read_from_csv(const string& filename, vector<unsigned int>& data) {
    ifstream file(filename);
    int number_of_rows = 0;
    long long int total_no_of_elements = 0;

    string line;
    while (getline(file, line))
    {
        number_of_rows++;

        string temp_cell;
        for (size_t i = 0; i <= line.size(); i++)
        {
            if (i == line.size() || line[i] == ',') {
                size_t start = temp_cell.find_first_not_of(" \t\r\n");
                size_t end   = temp_cell.find_last_not_of(" \t\r\n");
                if (start != string::npos) {
                    temp_cell = temp_cell.substr(start, end - start + 1);
                    int value = atoi(temp_cell.c_str());
                    data.push_back(value);
                    total_no_of_elements++;
                }
                temp_cell = "";
            } else {
                temp_cell += line[i];
            }
        }
    }
    file.close();

    // Error check
    if (data.empty()) {
        cout << "Error: No valid data in file: " << filename << endl;
    }

    int *dimensions = (int *)malloc(2 * sizeof(int));
    dimensions[0] = number_of_rows;
    dimensions[1] = total_no_of_elements;
    return dimensions;
}

unsigned int max_muller_finder(vector<unsigned int> &v) {

    unsigned int max_muller = 0;
    // max finding
    for (size_t i = 0; i < v.size(); i++)
    {
        if (v[i] > max_muller)
            max_muller = v[i];
    }
    return max_muller;
}

vector<unsigned int> cpu_counting_sort(vector<unsigned int> &v_1) {

    vector<unsigned int> v_2, v_c;
    v_2.resize(v_1.size());

    unsigned int max_muller = 0;
    // max finding
    for (size_t i = 0; i < v_1.size(); i++)
    {
        if (v_1[i] > max_muller)
            max_muller = v_1[i];
    }
    // cout << "setp 1 done" << endl;
    // cout << "max_muller: " << max_muller << endl;
    // axilliary array creation
    for (size_t i = 0; i <= max_muller; i++)
    {
        v_c.push_back(0);
    }
    // cout << "setp 2 done" << endl;
    for (size_t j = 0; j < v_1.size(); j++)
    {
        v_c[v_1[j]]++;
    }
    // cout << "setp 3 done" << endl;
    // prefix sum
    for (size_t i = 1; i <= max_muller; i++)
    {
        v_c[i] += v_c[i-1];
    }
    // cout << "setp 4 done" << endl;

    // last loop which needs to be in-place
    for (int j = v_1.size()-1; j >= 0; j--)
    {
        v_2[v_c[v_1[j]] - 1] = v_1[j];
        v_c[v_1[j]]--;
    }
    // cout << "step 5 done" << endl;
    return v_2;
}

__global__ void gpu_counting_sort(unsigned int *matrix_alike, int *row_offsets, int *row_sizes,
                                  unsigned int *sorted_matrix, unsigned int max_value, int total_flat_size) {
    
    int row_idx = blockIdx.x;
    int row_start = row_offsets[row_idx];
    int row_size = row_sizes[row_idx];

    // the following check should never return true, => redundant
    if (row_start + row_size + 1 > total_flat_size) {
        printf("security guard was true\n");
        return;
    }


    extern __shared__ unsigned int shared_mem[];
    unsigned int *count = shared_mem;

    // Step 1: Initialize count array
    for (int i = threadIdx.x; i <= max_value; i += blockDim.x)
        count[i] = 0;
    __syncthreads();

    // Step 2: Count frequencies
    for (int i = threadIdx.x; i < row_size; i += blockDim.x) {
        unsigned int value = matrix_alike[row_start + 1 + i];
        atomicAdd(&count[value], 1);
    }
    __syncthreads();

    // Step 3: Prefix sum (serial by threadIdx.x == 0)
    if (threadIdx.x == 0) {
        unsigned int total = 0;
        for (int i = 0; i <= max_value; i++) {
            unsigned int old_count = count[i];
            count[i] = total;
            total += old_count;
        }
    }
    __syncthreads();

    // Step 4: Deterministically write sorted result (single thread)
    if (threadIdx.x == 0) {
        for (int i = 0; i < row_size; i++) {
            unsigned int value = matrix_alike[row_start + 1 + i];
            unsigned int pos = count[value];
            sorted_matrix[row_start + 1 + pos] = value;
            count[value] += 1;
        }
    }
}

// alt kernel
__global__ void alt_gpu_counting_sort(unsigned int *matrix_alike, int *row_offsets, int *row_sizes,
                                  unsigned int *sorted_matrix, unsigned int max_value, int total_flat_size) {


    int row_idx = blockIdx.x;
    int row_start = row_offsets[row_idx];
    int row_size = row_sizes[row_idx];
    unsigned int indiv_elem = row_start + threadIdx.x;

    // if (threadIdx.x == 0 && blockIdx.x == 0) {
    //     printf("DEBUG KERNEL LAUNCHED total_flat_size=%d max_value=%u row_start=%d row_size=%d\n",
    //         total_flat_size, max_value, row_start, row_size);
    // }
    // __syncthreads();

    extern __shared__ unsigned int shared_mem[];
    unsigned int *count = shared_mem;
    // count array initialization
    count[threadIdx.x] = 0;
    __syncthreads();

    // trying to count frequancies
    if (threadIdx.x < row_size) {
        unsigned int indiv_elem = row_start + 1 + threadIdx.x; // +1 to skip row size
        atomicAdd(&count[matrix_alike[indiv_elem]], 1);
    }

    // trying prefix sum now
    // Serial prefix sum (more reliable for counting sort)
    if (threadIdx.x == 0) {
        // Convert counts to cumulative counts
        for (int i = 1; i <= max_value; i++) {
            count[i] += count[i-1];
        }
        
        // Convert to exclusive prefix sum (shift right)
        for (int i = max_value; i > 0; i--) {
            count[i] = count[i-1];
        }
        count[0] = 0;
    }
    __syncthreads();

    //DEBUG
    // printf("HELLO!\n");
    // if (row_idx == 0 && threadIdx.x == 0) {
    //     printf("HELLO!\n");
    //     for (int i=0; i<256; i++) {
    //         printf("%d ", count[i]);
    //     }
    //     printf("\n");
    // }

    // trying to place stored value now
    if (threadIdx.x < row_size) {
        unsigned int data_idx = row_start + 1 + threadIdx.x; // +1 to skip row size
        unsigned int value = matrix_alike[data_idx];
        
        // Get position and increment for next element with same value
        int pos = atomicAdd(&count[value], 1); // atomicAdd, not atomicSub!
        sorted_matrix[row_start + 1 + pos] = value;
    }
    
    // Copy the row size to output
    if (threadIdx.x == 0) {
        sorted_matrix[row_start] = matrix_alike[row_start];
    }
}

void write_matrix_to_csv(const string& filename, const vector<unsigned int>& mat, int no_of_real_elements, int no_of_rows_of_matrix) {

    ofstream file(filename);

    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << "\n";
        return;
    }

    int jmp = 0;
    for (size_t i = 0; i < no_of_real_elements + no_of_rows_of_matrix; i += (jmp+1)) {
        jmp = mat[i];
        for (size_t j = i; j<=i+jmp; j++) {
            file << mat[j];
            if (j < i + jmp) file << ",";
        }
        file << "\n";
    }

    // for (int i = 0; i < rows; i++) {
    //     for (int j = 0; j < cols; j++) {
    //         file << mat[i * cols + j];
    //         if (j < cols - 1) 
    //             file << ",";  // add comma except last element
    //     }
    //     file << "\n";
    // }

    file.close();
}


int main(int argc, char *argv[]) {

    if (argc != 4 && string(argv[2]) != "-o") {
        cout << "WRONG NUMBER OF ARGUMENTS!!" << endl;
        exit(0);
    }
    string name_of_ip_file = argv[1];
    string name_of_op_file = argv[3];

    vector<unsigned int> matrix_alike;

    // cout << "You want cpu multiply or gpu multiply?\n\
    // Press 0 for CPU, Press 1 for GPU" << endl;
    int c = 1;
    // cin >> c;
    // while (c!=0 && c!=1) {
    //     cout << "BAD CHOICE> ENTER AGAIN! >_<" << endl;
    //     cin >> c;
    //     // return 0;
    // }

    // GPU pointers
    unsigned int *d_matrix_alike;
    int *d_row_offsets, *d_row_sizes;
    unsigned int *d_sorted_matrix;


    int *dimensions_1 = read_from_csv(name_of_ip_file, matrix_alike);
    int no_of_rows_of_matrix = dimensions_1[0];
    int no_of_real_elements = dimensions_1[1] - dimensions_1[0];

    cout << "no_of_rows_of_matrix: " << no_of_rows_of_matrix << endl;
    cout << "no_of_real_elements: " << no_of_real_elements << endl;

    // // NORMAL PRINTING
    // for (int i=0; i<matrix_alike.size(); i++) {
    //     cout << matrix_alike.at(i) << " ";
    // }
    // cout << endl;

    // // PREETY Printing:
    int jmp = 0;
    // for (size_t i = 0; i < no_of_real_elements + no_of_rows_of_matrix; i += (jmp+1))
    // {
    //     jmp = matrix_alike[i];
    //     for (size_t j = i; j <= i+jmp; j++) {
    //         cout << matrix_alike[j] << " ";
    //     }
    //     cout << endl;
    // }
    // return 0;
    
    vector<unsigned int> sorted;
    // CPU sorting call
    if (c==0) {
        // sorted = counting_sort(matrix_alike);

        jmp = 0;
        for (int i=0; i<no_of_real_elements+no_of_rows_of_matrix; i+=(jmp+1)) {
            jmp = matrix_alike[i];
            vector<unsigned int> inter;
            for (size_t j = i+1; j <= i+jmp; j++) {
                inter.push_back(matrix_alike[j]);
            }
            sorted = cpu_counting_sort(inter);

            int off = 0;
            for (size_t j = i+1; j <= i+jmp; j++) {
                matrix_alike[j] = sorted[off];
                off++;
            }
        }
        cout << "+++++++++++++++++++" << endl;
        // // PREETY Printing:
        // jmp = 0;
        // for (size_t i = 0; i < no_of_real_elements + no_of_rows_of_matrix; i += (jmp+1))
        // {
        //     jmp = matrix_alike[i];
        //     for (size_t j = i+1; j <= i+jmp; j++) {
        //         cout << matrix_alike[j] << " ";
        //     }
        //     cout << endl;
        // }

        // Write output to out file:
        write_matrix_to_csv(name_of_op_file, matrix_alike, no_of_real_elements, no_of_rows_of_matrix);
    }
    else if (c==1) {
        // finding max value in the whole matrix:
        unsigned int max_muller = max_muller_finder(matrix_alike);

        // Offset calculations:
        vector<int> row_offsets;
        vector<int> row_sizes;
        int offset = 0;
        while (offset < matrix_alike.size()) {
            row_offsets.push_back(offset);             // The position of the num_elements
            int row_size = matrix_alike[offset];      // Read the row size
            row_sizes.push_back(row_size);
            offset += 1 + row_size;                   // Move to the next row
        }

        // // printing row_offsets & row_sizes
        // cout << "row_offsets" << endl;
        // for (size_t i = 0; i < row_offsets.size(); i++)
        // {
        //     cout << row_offsets[i] << " ";
        // }
        // cout << endl;
        // cout << "row_sizes" << endl;
        // for (size_t i = 0; i < row_sizes.size(); i++)
        // {
        //     cout << row_sizes[i] << " ";
        // }
        // cout << endl;
        // cout << "entire size: " << matrix_alike.size() << endl;

        // Memory allocation & transfer:
        cudaMalloc(&d_matrix_alike, matrix_alike.size() * sizeof(unsigned int));
        cudaMalloc(&d_row_offsets, row_offsets.size() * sizeof(int));
        cudaMalloc(&d_row_sizes, row_sizes.size() * sizeof(int));
        cudaMalloc(&d_sorted_matrix, matrix_alike.size() * sizeof(unsigned int));

        cudaMemcpy(d_matrix_alike, matrix_alike.data(), matrix_alike.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_offsets, row_offsets.data(), no_of_rows_of_matrix * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_sizes, row_sizes.data(), no_of_rows_of_matrix * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_sorted_matrix, d_matrix_alike, matrix_alike.size() * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

        // Kernel Launch config:
        // int threads_per_block = 256;
        size_t shared_mem_size = (max_muller + 1) * sizeof(unsigned int);

        //timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);

        alt_gpu_counting_sort<<<no_of_rows_of_matrix, 256, 256*sizeof(int)>>>(
            d_matrix_alike, d_row_offsets, d_row_sizes, d_sorted_matrix, max_muller, matrix_alike.size());
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            fprintf(stderr, "Launch error: %s\n", cudaGetErrorString(err));
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            fprintf(stderr, "Sync error (kernel died): %s\n", cudaGetErrorString(err));
        }
        // closing the timing counter
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start, stop);

        printf("Kernel elapsed time: %.3f microseconds\n", ms * 1000);

        // Copying results back to host:
        vector<unsigned int> sorted_host(matrix_alike.size());
        cudaMemcpy(sorted_host.data(), d_sorted_matrix, matrix_alike.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        cout << "============================" << endl;
        // PRETTY PRINTING
        // jmp = 0;
        // for (size_t i = 0; i < no_of_real_elements + no_of_rows_of_matrix; i += (jmp+1))
        // {
        //     jmp = sorted_host[i];
        //     for (size_t j = i; j <= i+jmp; j++) {
        //         cout << sorted_host[j] << " ";
        //     }
        //     cout << endl;
        // }

        // Write output to out file:
        write_matrix_to_csv(name_of_op_file, sorted_host, no_of_real_elements, no_of_rows_of_matrix);
    }
    
    // memory freeing:
    free(dimensions_1);
    cudaFree(d_matrix_alike);
    cudaFree(d_row_offsets);
    cudaFree(d_row_sizes);
    cudaFree(d_sorted_matrix);
    return 0;
}