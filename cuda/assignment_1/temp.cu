__global__ void gpu_mat_mul(int *d_matrix_1, int *d_matrix_2, int *d_matrix_res, int m, int n, int k) {

    unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;

    unsigned int i = blockIdx.x;
    unsigned j = threadIdx.x;

    for (size_t i = 0; i < m; i++)
    {
        for (size_t j = 0; j < k; j++)
        {
            for (size_t l = 0; l < n; l++)
            {
                d_matrix_res[id] += (d_matrix_1[i*n + l] * d_matrix_2[l*k + j]);
            }
        }
    }
}