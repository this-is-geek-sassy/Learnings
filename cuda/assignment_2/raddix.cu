#include <bits/stdc++.h>
#include <cuda.h>

using namespace std;

int *read_from_csv(const string& filename, vector<unsigned int>& data) {
    ifstream file(filename);
    long long int no_of_elements = 0;
    long long int total_no_of_elements = 0;

    string line;
    int number_of_rows = 0;

    while (getline(file, line))
    {
        total_no_of_elements += no_of_elements;
        no_of_elements = 0;
        
        // First, count how many elements are in this line
        string temp_line = line;
        string temp_cell;
        for (size_t i = 0; i <= temp_line.size(); i++)
        {
            if (i == temp_line.size() || temp_line[i] == ',') {
                size_t start = temp_cell.find_first_not_of(" \t\r\n");
                size_t end   = temp_cell.find_last_not_of(" \t\r\n");
                if (start != string::npos) {
                    no_of_elements++;
                }
                temp_cell = "";
            } else {
                temp_cell += temp_line[i];
            }
        }
        
        // Add the count at the beginning of this row
        data.push_back(no_of_elements);
        
        // Now process the actual data
        string cell;
        for (size_t i = 0; i <= line.size(); i++)
        {
            if (i == line.size() || line[i] == ',') {
                
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
    total_no_of_elements += no_of_elements;
    file.close();

    // Error check
    if (data.empty()) {
        cout << "Error: No valid data in file: " << filename << endl;
    }

    // Return dimensions
    int *dimensions = (int *)malloc(2 * sizeof(int));
    dimensions[0] = number_of_rows;
    dimensions[1] = total_no_of_elements; // Not needed as per your requirement
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

    if (row_start + row_size + 1 > total_flat_size)
        return;

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


int main(int argc, char *argv[]) {

    if (argc != 4 && string(argv[2]) != "-o") {
        cout << "WRONG NUMBER OF ARGUMENTS!!" << endl;
        exit(0);
    }
    string name_of_ip_file = argv[1];
    string name_of_op_file = argv[3];

    vector<unsigned int> matrix_alike;

    cout << "You want cpu multiply or gpu multiply?\n\
    Press 0 for CPU, Press 1 for GPU" << endl;
    int c = -99;
    cin >> c;
    while (c!=0 && c!=1) {
        cout << "BAD CHOICE> ENTER AGAIN! >_<" << endl;
        cin >> c;
        // return 0;
    }

    // GPU pointers
    unsigned int *d_matrix_alike;
    int *d_row_offsets, *d_row_sizes;
    unsigned int *d_sorted_matrix;


    int *dimensions_1 = read_from_csv(name_of_ip_file, matrix_alike);
    int no_of_rows_of_matrix = dimensions_1[0];
    int no_of_real_elements = dimensions_1[1];

    cout << "no_of_rows_of_matrix: " << no_of_rows_of_matrix << endl;
    cout << "no_of_real_elements: " << no_of_real_elements << endl;

    // // NORMAL PRINTING
    // for (int i=0; i<matrix_alike.size(); i++) {
    //     cout << matrix_alike.at(i) << " ";
    // }
    // cout << endl;

    // PREETY Printing:
    int jmp = 0;
    for (size_t i = 0; i < no_of_real_elements + no_of_rows_of_matrix; i += (jmp+1))
    {
        jmp = matrix_alike[i];
        for (size_t j = i+1; j <= i+jmp; j++) {
            cout << matrix_alike[j] << " ";
        }
        cout << endl;
    }
    
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
        // PREETY Printing:
        jmp = 0;
        for (size_t i = 0; i < no_of_real_elements + no_of_rows_of_matrix; i += (jmp+1))
        {
            jmp = matrix_alike[i];
            for (size_t j = i+1; j <= i+jmp; j++) {
                cout << matrix_alike[j] << " ";
            }
            cout << endl;
        }
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

        // Memory transfer:
        cudaMalloc(&d_matrix_alike, matrix_alike.size() * sizeof(unsigned int));
        cudaMalloc(&d_row_offsets, row_offsets.size() * sizeof(int));
        cudaMalloc(&d_row_sizes, row_sizes.size() * sizeof(int));
        cudaMalloc(&d_sorted_matrix, matrix_alike.size() * sizeof(unsigned int));

        cudaMemcpy(d_matrix_alike, matrix_alike.data(), matrix_alike.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_offsets, row_offsets.data(), no_of_rows_of_matrix * sizeof(int), cudaMemcpyHostToDevice);
        cudaMemcpy(d_row_sizes, row_sizes.data(), no_of_rows_of_matrix * sizeof(int), cudaMemcpyHostToDevice);

        // Copy input to output so that row sizes are preserved
        cudaMemcpy(d_sorted_matrix, d_matrix_alike, matrix_alike.size() * sizeof(unsigned int), cudaMemcpyDeviceToDevice);

        // Kernel Launch config:
        int threads_per_block = 256;
        size_t shared_mem_size = (max_muller + 1) * sizeof(unsigned int);

        gpu_counting_sort<<<no_of_rows_of_matrix, 256, shared_mem_size>>>(
            d_matrix_alike, d_row_offsets, d_row_sizes, d_sorted_matrix, max_muller, matrix_alike.size());
        cudaDeviceSynchronize();

        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("CUDA error: %s\n", cudaGetErrorString(err));
        }

        // Copying results back to host:
        vector<unsigned int> sorted_host(matrix_alike.size());
        cudaMemcpy(sorted_host.data(), d_sorted_matrix, matrix_alike.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        // PRETTY PRINTING
        cout << "============================" << endl;
        jmp = 0;
        for (size_t i = 0; i < no_of_real_elements + no_of_rows_of_matrix; i += (jmp+1))
        {
            jmp = sorted_host[i];
            for (size_t j = i+1; j <= i+jmp; j++) {
                cout << sorted_host[j] << " ";
            }
            cout << endl;
        }
    }
    
    // memory freeing:
    free(dimensions_1);
    cudaFree(d_matrix_alike);
    cudaFree(d_row_offsets);
    cudaFree(d_row_sizes);
    return 0;
}