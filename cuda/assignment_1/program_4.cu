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
std::vector<int> tile_transpose(const std::vector<int>& mat, int rows, int cols, int tile_D1, int tile_D2) {
    std::vector<int> transposed(cols * rows);

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