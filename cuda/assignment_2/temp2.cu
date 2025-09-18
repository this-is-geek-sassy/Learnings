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
