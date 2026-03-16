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

    // Initialize K and P to the frontier of the diagonal
    if (i > cardA) {
        Kx = i-cardA;
        Ky = cardA;
        Px = cardA;
        Py = i-cardA;
    }
    else {
        
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

    // Initialize K and P to the frontier of the diagonal
    if (tidx > cardA) {
        Kx = tidx-cardA;
        Ky = cardA;
        Px = cardA;
        Py = tidx-cardA;
    }
    else {
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

__global__ void sortSmallBatch_k(float* Ms, int N, int d) {
    /* Sorts each array in Ms
    - Ms: contains the N arrays of length d, stored contiguously
    - N: The number of arrays to sort
    - d: The length of each array, must be <= 1024 due to the fact that max 1024 threads can be launched by a single block 

    Each block will be in charge of sorting one array.
    */

    int list_idx = blockIdx.x; // The list processed by this thread will be Ms[list_idx * d : list_idx * d + d]
    if (list_idx >= N) return; // This block is useless

    float* M = &Ms[list_idx * d];
    int tidx_in_M = threadIdx.x; // The index of this thread in the list M
    if (tidx_in_M >= d) return; //This thread is useless

    // Create two shared buffers to avoid reading and writing at the same location and to read/write faster
    extern __shared__ float shared[]; 
    float* bufA = shared;
    float* bufB = shared + d;
    float* input = bufA;
    float* output = bufB;

    bufA[tidx_in_M] = M[tidx_in_M];
    __syncthreads();


    for (int step = 1 ; step < d ; step *= 2){
        int merge_idx = tidx_in_M / (step * 2); // index of the merge this thread will participate to
        int base = 2 * step * merge_idx;
        float* A = &input[base];
        float* B = &input[base + step]; // The two lists to merge
        int i = tidx_in_M % (2 * step);

        // Skip threads that would write outside the array
        bool active = (base + i  < d);
        

        // Compute real sizes of A and B
        int sizeA = min(step, d - base);
        int sizeB = min(step, max(0, d - (base + step)));

        int Kx, Ky, Px, Py, Qx, Qy;
        int offset;

        if (active) {
            // Initialize K and P to the frontier of the diagonal
            if (i > sizeA) {
                Kx = i - sizeA;
                Ky = sizeA;
                Px = sizeA;
                Py = i - sizeA;
            }
            else {
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
        
                
                if (Qy >= 0 && Qx <= sizeB && (Qy == sizeA || Qx == 0 || A[Qy] > B[Qx-1])) {
                    if (Qx == sizeB || Qy == 0 || A[Qy-1] <= B[Qx]) {
                        // We cornered the solution, we can return the value of M[i]
                        if (Qy < sizeA && (Qx == sizeB || A[Qy] <= B[Qx])) {
                            output[base + i] = A[Qy];
                        }
                        else {
                            output[base + i] = B[Qx];
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
        
        __syncthreads();

        // Swap buffers
        float* temp = input;
        input = output;
        output = temp;

        __syncthreads();
    }

    M[tidx_in_M] = input[tidx_in_M];
    

}

__global__ void mergeBatch_k(float* ABs, float* Ms, int* cardAs, int N, int d) {
    /* for each sorted A and B of As and Bs, merge them into the corresponding M of Ms.
    Arguments :
     - ABs : input sorted lists, containing 2N lists. Each pair of list A and B is stored contiguously, that is to say A[i] = ABs[d * i : d * i + cardAs[i]] and B[i] = ABs[d * i + cardAs[i] : d * i + cardAs[i] + cardBs[i]]
     - Ms : output merged lists, containing N lists of fixed size 2d.
     - cardAs : list of the cardinals of the A lists (cardBs can be deducted since d = |As[i]| + |Bs[i]|)
     - N : number of lists to merge.
     - d : d = |As[i]| + |Bs[i]| for each i < N

     The only difference with mergeSmallBatch_k is that this kernel can merge lists that are longer than 1024, because it distributes the workload across multiple blocks.
    */


    int gbx = threadIdx.x + blockDim.x * blockIdx.x; // Here we need to use the global index of the thread
    int list_idx = gbx / d; // The index of the list that this thread will participate in merging
    if (list_idx >= N) return; // If the list index is out of bounds, it means that this thread is useless in our scenario

    float* A = &ABs[list_idx * d];
    int cardA = cardAs[list_idx];
    int cardB = d - cardA;
    float* B = &ABs[list_idx * d + cardA];
    float* M = &Ms[list_idx * d];
    int tidx = gbx - d * list_idx; // The index of the item that this thread will merge in M

    int Kx, Ky, Px, Py, Qx, Qy;
    int offset;
    
    // Initialize K and P to the frontier of the diagonal
    if (tidx > cardA) {
        Kx = tidx-cardA;
        Ky = cardA;
        Px = cardA;
        Py = tidx-cardA;
    }
    else {
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
                    M[tidx] = A[Qy];
                }
                else {
                    M[tidx] = B[Qx];
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

    //***********************//
    //// TEST EXERCICE 2.a ////
    //***********************//
    {
        const int N = 100000;
        int d_vals[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
        int num_d = 10;

        FILE* csv = fopen("benchmark2a.csv", "w");
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
        printf("Benchmark results saved to benchmark2a.csv\n");
    }

    //***********************//
    //// TEST EXERCICE 2.b ////
    //***********************//
    {
        FILE* csv = fopen("benchmark2b.csv", "w");
        fprintf(csv, "d,N,time_ms\n");
        const int N = 100000;
        int d_vals[] = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
        int num_d = 10;

        for (int di = 0; di < num_d; di++) {
            int d = d_vals[di];

        float* h_Ms   = (float*)malloc(N * d * sizeof(float));
        
        for (int i = 0; i < N * d; i++) {
            h_Ms[i] = (float)(rand() % 100);
        }

        // Can be used to check that the lists are correctly merged by printing them
        // printf("Lists before sorting:\n");
        // for (int i = 0; i < N ; i++){
        //     for (int j = 0 ; j < d ; j++){
        //         printf("%.0f ", h_Ms[i * d + j]);
        //     }
        //     printf("\n");
        // }

        float* d_Ms;
        testCUDA(cudaMalloc((void**)&d_Ms,     N * d * sizeof(float)));
        testCUDA(cudaMemcpy(d_Ms,    h_Ms,    N * d * sizeof(float), cudaMemcpyHostToDevice));

        // Warmup
        sortSmallBatch_k<<<N, d, 2 * d*sizeof(float)>>>(d_Ms, N, d);
        testCUDA(cudaDeviceSynchronize());

        // Timed run
        cudaEvent_t start, stop;
        float elapsed_ms = 0.0f;
        testCUDA(cudaEventCreate(&start));
        testCUDA(cudaEventCreate(&stop));
        testCUDA(cudaEventRecord(start));
        sortSmallBatch_k<<<N, d, 2 * d * sizeof(float)>>>(d_Ms, N, d); // Here we need to launch the kernel with shared memory
        testCUDA(cudaGetLastError());
        testCUDA(cudaDeviceSynchronize());
        testCUDA(cudaEventRecord(stop));
        testCUDA(cudaEventSynchronize(stop));
        testCUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));

        testCUDA(cudaMemcpy(h_Ms, d_Ms, N * d * sizeof(float), cudaMemcpyDeviceToHost));

        // Can be used to check that the lists are correctly merged by printing them
        // printf("Lists after sorting:\n");
        // for (int i = 0; i < N ; i++){
        //     for (int j = 0 ; j < d ; j++){
        //         printf("%.0f ", h_Ms[i * d + j]);
        //     }
        //     printf("\n");
        // }

        printf("d=%4d | N=%d | time=%.4f ms\n", d, N, elapsed_ms);
        fprintf(csv, "%d,%d,%.6f\n", d, N, elapsed_ms);


        testCUDA(cudaEventDestroy(start));
        testCUDA(cudaEventDestroy(stop));
        cudaFree(d_Ms);
        }

        fclose(csv);
        printf("Benchmark results saved to benchmark2b.csv\n");
        
    }

    //*********************//
    //// TEST EXERCICE 3 ////
    //*********************//
    {
        const int N = 100000;
        int d_vals[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};
        int num_d = 14;

        FILE* csv = fopen("benchmark3.csv", "w");
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

            // Can be used to check that the lists are correctly merged by printing them
            // printf("Lists before merging:\n");
            // for (int i = 0; i < N; i++){
            //     printf("A: ");
            //     for (int j = 0 ; j < h_cardAs[i] ; j++) {
            //         printf("%.0f ", h_ABs[i * d + j]);
            //     }
            //     printf("\nB: ");
            //     for (int j = h_cardAs[i] ; j < d ; j++) {
            //         printf("%.0f ", h_ABs[i * d + j]);
            //     }
            //     printf("\n");
            // }

            float* d_ABs;
            float* d_Ms;
            int*   d_cardAs;
            testCUDA(cudaMalloc((void**)&d_ABs,    N * d * sizeof(float)));
            testCUDA(cudaMalloc((void**)&d_Ms,     N * d * sizeof(float)));
            testCUDA(cudaMalloc((void**)&d_cardAs, N * sizeof(int)));
            testCUDA(cudaMemcpy(d_ABs,    h_ABs,    N * d * sizeof(float), cudaMemcpyHostToDevice));
            testCUDA(cudaMemcpy(d_cardAs, h_cardAs, N * sizeof(int),       cudaMemcpyHostToDevice));

            int num_blocks = 1 + N * d / 1024;

            // Warmup
            mergeBatch_k<<<num_blocks, 1024>>>(d_ABs, d_Ms, d_cardAs, N, d);
            testCUDA(cudaDeviceSynchronize());

            // Timed run
            cudaEvent_t start, stop;
            float elapsed_ms = 0.0f;
            testCUDA(cudaEventCreate(&start));
            testCUDA(cudaEventCreate(&stop));
            testCUDA(cudaEventRecord(start));
            mergeBatch_k<<<num_blocks, 1024>>>(d_ABs, d_Ms, d_cardAs, N, d);
            testCUDA(cudaEventRecord(stop));
            testCUDA(cudaEventSynchronize(stop));
            testCUDA(cudaEventElapsedTime(&elapsed_ms, start, stop));
            
            float* h_Ms   = (float*)malloc(N * d * sizeof(float));
            testCUDA(cudaMemcpy(h_Ms,    d_Ms,    N * d * sizeof(float), cudaMemcpyDeviceToHost));

            // Can be used to check that the lists are correctly sorted by printing them
            // printf("Lists after merging:\n");
            // for (int i = 0; i < N ; i++){
            //     for (int j = 0 ; j < d ; j++){
            //         printf("%.0f ", h_Ms[i * d + j]);
            //     }
            //     printf("\n");
            // }

            // Checks that every list has been correctly merged (without printing all the lists)
            // for (int i = 0 ; i < d * N ; i++) {
            //     if (i % d == 0) continue;
            //     if (h_Ms[i-1] > h_Ms[i]) {
            //         printf("The list is not sorted\n");
            //     }
            // }

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
        printf("Benchmark results saved to benchmark3.csv\n");

        

    }

	return 0;
}