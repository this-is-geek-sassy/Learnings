#include <bits/stdc++.h>
#include <cuda.h>

__global__ void printk(int *counter) {
    ++*counter; // in general, this can be arbitrary processing
    printf("\t%d\n", *counter);
}

int main() {
    int hcounter = 0, *counter;

    cudaHostAlloc(&counter, sizeof(int), 0);

    do {
        printf("host: %d\n", *counter);
        // cudaMemcpy(counter, &hcounter, sizeof(int), cudaMemcpyHostToDevice);

        printk<<<1,1>>>(counter);
        cudaError_t err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
        ++*counter;
        // cudaMemcpy(&hcounter, counter, sizeof(int), cudaMemcpyDeviceToHost);
    } while(*counter < 10);         // in general, this can be arbitrary processing

    cudaDeviceSynchronize();  // Final sync to flush all output
    cudaFree(counter);
    return 0;
}