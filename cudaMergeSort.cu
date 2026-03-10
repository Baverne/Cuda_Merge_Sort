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
    /* Merge sorted lists A and B into a sorted merged M.
    Use one thread i for each M[i]
    Arguments :
     - A, B : input sorted lists
     - M : output merged list
     - cardA, cardB : sizes of A and B
    */
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

__global__ void mergeSmallBatch_k(float* ABs, float* Ms, int* cardAs, int N, int d) {
    /* for each sorted A and B of As and Bs, merge them into the corresponding M of Ms.
    Arguments :
     - ABs : input sorted lists, containing 2N lists. Each pair of list A and B is stored contiguously, that is to say A[i] = ABs[d * i : d * i + cardAs[i]] and B[i] = ABs[d * i + cardAs[i] : d * i + cardAs[i] + cardBs[i]]
     - Ms : output merged lists, containing N lists of fixed size 2d.
     - cardAs : list of the cardinals of the A lists (cardBs can be deducted since d = |As[i]| + |Bs[i]|)
     - N : number of lists to merge.
     - d : d = |As[i]| + |Bs[i]| for each i < N
    */
    
    int Qt = threadIdx.x/d; // In this thread, we are going to deal with the list M = Ms[blockIdx.x*(blockDim.x/d) + Qt : blockIdx.x * blockDim.x + Qt + d]
    int gbx = Qt + blockIdx.x*(blockDim.x/d); // That is to say M = Ms[gbx : gbx + d]. gbx is the index of the list that we are dealing with.
    if (gbx >= N) return; // So no point to continue if we are out of bound
    int tidx = threadIdx.x - (Qt*d); // More precisely, In this thread, we are going to fill in M[i] = Ms[gbx + tidx], tidx id the index of the digit we are dealing with within the correst list.
    
    
    /* at this point, we re-writte the code of the previous kernel, but with the following changes :
    - cardA = cardAs[gbx]
    - cardB = d - cardA
    - A = ABs[d * gbx : d * gbx + cardAs[gbx]]
    - B = ABs[d * gbx + cardAs[gbx] : d * gbx + cardAs[gbx] + cardB]
    - M[i] <- M[gbx + tidx]
    */

    int cardA = cardAs[gbx];
    int cardB = d - cardA;
    float* A = &ABs[gbx * d];
    float* B = &ABs[gbx * d + cardA];
    int Kx, Ky, Px, Py, Qx, Qy;
    int offset;

    if (tidx > cardA) {
        // PAS SUR DE COMPRENDRE CETTE ETAPE %%
        Kx = tidx-cardA;
        Ky = cardA;
        Px = cardA;
        Py = tidx-cardA;
    }
    else {
        // In this case we initialize the K and P to the frontier of the diagonal
        Kx = 0;
        Ky = tidx;
        Px = tidx;
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
                    Ms[gbx * d + tidx] = A[Qy];
                }
                else {
                    Ms[gbx * d + tidx] = B[Qx];
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
    //*********************//
    //// TEST EXERCICE 1 ////
    //*********************//
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

    //*********************//
    //// TEST EXERCICE 2 ////
    //*********************//
    {
        const int N = 100000;
        int d_vals[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
        int num_d = 10;

        FILE* csv = fopen("benchmark.csv", "w");
        fprintf(csv, "d,N,time_ms\n");

        for (int di = 0; di < num_d; di++) {
            // for each d, we will measure sorting time
            int d = d_vals[di];
            int cardA_val = d / 2; // longest merge is when both list are equal size
            int cardB_val = d - cardA_val;

            // Build host ABs: each pair i = sorted A (cardA_val floats) then sorted B (cardB_val floats)
            float* h_ABs   = (float*)malloc(N * d * sizeof(float));
            int*   h_cardAs = (int*)malloc(N * sizeof(int));
            
            // In order to build sorted list of a given size, we add a random value between 0 and 9 to the previous one
            for (int i = 0; i < N; i++) {
                h_cardAs[i] = cardA_val;
                float val = 0.0f;
                for (int j = 0; j < cardA_val; j++) {
                    val += (float)(rand() % 10);
                    h_ABs[i * d + j] = val;
                }
                val = 0.0f;
                for (int j = 0; j < cardB_val; j++) {
                    val += (float)(rand() % 10);
                    h_ABs[i * d + cardA_val + j] = val;
                }
            }

            float* d_ABs;
            float* d_Ms;
            int*   d_cardAs;
            testCUDA(cudaMalloc((void**)&d_ABs,    N * d * sizeof(float)));
            testCUDA(cudaMalloc((void**)&d_Ms,     N * d * sizeof(float)));
            testCUDA(cudaMalloc((void**)&d_cardAs, N * sizeof(int)));
            testCUDA(cudaMemcpy(d_ABs,    h_ABs,    N * d * sizeof(float), cudaMemcpyHostToDevice));
            testCUDA(cudaMemcpy(d_cardAs, h_cardAs, N * sizeof(int),       cudaMemcpyHostToDevice));

            int lists_per_block = 1024 / d;
            int num_blocks      = (N + lists_per_block - 1) / lists_per_block;

            // Warmup
            mergeSmallBatch_k<<<num_blocks, 1024>>>(d_ABs, d_Ms, d_cardAs, N, d);
            testCUDA(cudaDeviceSynchronize());

            // Timed run
            cudaEvent_t start, stop;
            float elapsed_ms = 0.0f;
            testCUDA(cudaEventCreate(&start));
            testCUDA(cudaEventCreate(&stop));
            testCUDA(cudaEventRecord(start));
            mergeSmallBatch_k<<<num_blocks, 1024>>>(d_ABs, d_Ms, d_cardAs, N, d);
            testCUDA(cudaEventRecord(stop));
            testCUDA(cudaEventSynchronize(stop));
            testCUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

            printf("d=%4d | N=%d | time=%.4f ms\n", d, N, elapsed_ms);
            fprintf(csv, "%d,%d,%.6f\n", d, N, elapsed_ms);

            testCUDA(cudaEventDestroy(start));
            testCUDA(cudaEventDestroy(stop));
            cudaFree(d_ABs);
            cudaFree(d_Ms);
            cudaFree(d_cardAs);
            free(h_ABs);
            free(h_cardAs);
        }

        fclose(csv);
        printf("Benchmark results saved to benchmark.csv\n");
    }

	return 0;
}