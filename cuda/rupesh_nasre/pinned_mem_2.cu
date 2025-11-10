#include <bits/stdc++.h>
#include <cuda.h>

__global__ void printk(int *counter) {

    for (int i=1; *counter<10; i++) {
        if (*counter % 2) {
            ++*counter; // in general, this can be arbitrary processing
            printf("GPU: %d\n", *counter);
        }
    }
}

int main() {
    int hcounter = 0, *counter;

    cudaHostAlloc(&counter, sizeof(int), 0);

    printf("host: %d\n", *counter);
    // cudaMemcpy(counter, &hcounter, sizeof(int), cudaMemcpyHostToDevice);

    // ++*counter;
    printk<<<1,1>>>(counter);

    // cudaDeviceSynchronize();
    // cudaError_t err = cudaDeviceSynchronize();
    // if (err != cudaSuccess) {
    //     printf("CUDA Error: %s\n", cudaGetErrorString(err));
    // }
    for (int i=1; *counter<10; i++) {
        if (*counter % 2 == 0 && *counter < 10) {
            ++*counter;
            printf("host: %d\n", *counter);
        }
    }
    // cudaMemcpy(&hcounter, counter, sizeof(int), cudaMemcpyDeviceToHost);


    cudaDeviceSynchronize();  // Final sync to flush all output
    cudaFree(counter);
    return 0;
}