#include <stdio.h>

// Function that catches the error 
void testCUDA(cudaError_t error, const char* file, int line) {

	if (error != cudaSuccess) {
		printf("There is an error in file %s at line %d\n", file, line);
		exit(EXIT_FAILURE);
	}
}

// Has to be defined in the compilation in order to get the correct value of the 
// macros __FILE__ and __LINE__
#define testCUDA(error) (testCUDA(error, __FILE__ , __LINE__))

__global__ void mergeSmall_k(float* A, float*B, float*M, int cardA, int cardB) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int Kx, Ky, Px, Py, Qx, Qy;
    int offset;
    if (i > cardA) {
        // PAS SUR DE COMPRENDRE CETTE ETAPE %%
        Kx = i-cardA;
        Ky = cardA;
        Px = cardA;
        Py = i-cardA;
    }
    else {
        // In this case we initialize the K and P to the frontier of the diagonal
        Kx = 0;
        Ky = i;
        Px = i;
        Py = 0;
    }
    while (true) {
        // We proceed by dichotomy, offset is therefore the half of the distance between the two frontiers
        offset = abs(Ky-Py)/2;
        Qx = Kx + offset;
        Qy = Ky - offset;

        if (Qy >= 0 && Qx <= cardB && (Qy == cardA || Qx == 0 || A[Qy] > B[Qx-1])) {
            if (Qx == cardB || Qy == 0 || A[Qy-1] <= B[Qx]) {
                // We cornered the solution, we can return the value of M[i]
                if (Qy < cardA && (Qx == cardB || A[Qy] <= B[Qx])) {
                    M[i] = A[Qy];
                }
                else {
                    M[i] = B[Qx];
                }
                break;
            }
            else {
                // In this case our solution is in the upper right part of the diagonal
                Kx = Qx + 1;
                Ky = Qy - 1;
            }
        }
        else {
            // In this case our solution is in the lower left part of the diagonal
            Px = Qx - 1;
            Py = Qy + 1;
        }
        

    }
}

int main(void) {

    // Two sorted input arrays
    int cardA = 10, cardB = 13;
    int cardM = cardA + cardB;

    float h_A[] = { 1.0f,  3.0f,  5.0f,  8.0f, 8.0f,
                   17.0f, 21.0f, 26.0f, 31.0f, 40.0f };

    float h_B[] = { 2.0f,  4.0f,  6.0f,  7.0f,  9.0f,
                   11.0f, 14.0f, 18.0f, 22.0f, 28.0f,
                   35.0f, 42.0f, 50.0f };

    float h_M[23];

    // Device allocations
    float *d_A, *d_B, *d_M;
    testCUDA(cudaMalloc((void**)&d_A, cardA * sizeof(float)));
    testCUDA(cudaMalloc((void**)&d_B, cardB * sizeof(float)));
    testCUDA(cudaMalloc((void**)&d_M, cardM * sizeof(float)));

    // Copy inputs to device
    testCUDA(cudaMemcpy(d_A, h_A, cardA * sizeof(float), cudaMemcpyHostToDevice));
    testCUDA(cudaMemcpy(d_B, h_B, cardB * sizeof(float), cudaMemcpyHostToDevice));

    // Launch kernel: 1 block, cardM threads (23 <= 1024)
    mergeSmall_k<<<1, cardM>>>(d_A, d_B, d_M, cardA, cardB);
    testCUDA(cudaGetLastError());
    testCUDA(cudaDeviceSynchronize());

    // Copy result back
    testCUDA(cudaMemcpy(h_M, d_M, cardM * sizeof(float), cudaMemcpyDeviceToHost));

    // Print inputs and result
    printf("A (%d elements): ", cardA);
    for (int i = 0; i < cardA; i++) printf("%.0f ", h_A[i]);

    printf("\nB (%d elements): ", cardB);
    for (int i = 0; i < cardB; i++) printf("%.0f ", h_B[i]);

    printf("\nMerged (%d elements): ", cardM);
    for (int i = 0; i < cardM; i++) printf("%.0f ", h_M[i]);
    printf("\n");

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_M);

	return 0;
}