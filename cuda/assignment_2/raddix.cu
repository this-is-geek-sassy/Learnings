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


    // memory freeing:
    free(dimensions_1);
    return 0;
}