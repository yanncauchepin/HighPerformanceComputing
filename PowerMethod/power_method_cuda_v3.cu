// power_method_cuda_v2.cu
// Compiler : nvcc -o power_method_cuda_v2 power_method_cuda_v2.cu -lm
// Run : ./power_method_cuda_v2 Input/input_test_1000.txt

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <cuda_runtime.h>

#define ERROR_THRESHOLD 1e-5

// Number of blocks : gridDim.x
// Index in block loops : blockIdx.x
// Number of thread per block : blockDim.x
// Index in threads loops : threadIdx.x

double gettimeofday_ () {
    struct timeval time ;
    gettimeofday (&time, NULL) ;
    return time.tv_sec + (time.tv_usec * 1.0e-6L);
}

double* build_flattened_matrix (int size_row, int size_col) {
    double* matrix = (double*) malloc (size_row*size_col * sizeof(double)) ;
    return matrix ;
}

void free_flattened_matrix (double* matrix) {
    free(matrix) ;
}

int set_size_from_file (FILE* file) {
    int size;
    if (fscanf(file, "%d", &size) != 1) {
        fprintf(stderr, "Error reading the size of the matrix in the file.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    return size;
}

double* set_flattened_matrix_from_file (FILE* file, int size_row, int size_col) {
    double* matrix = build_flattened_matrix (size_row, size_col) ;
    for (int i = 0; i < size_row ; i++) {
        for (int j = 0; j < size_col ; j++) {
            if (fscanf(file, "%lf ", &matrix[i*size_col+j]) != 1) {
                fprintf(stderr, "Error reading the element [%d][%d] of the matrix in the file.\n", i, j);
                fclose(file);
                free_flattened_matrix(matrix);
                exit(EXIT_FAILURE);
            }
        }
    }
    return matrix;
}

void print_flattened_matrix (double* matrix, int size_row, int size_col) {
    for (int i=0 ; i<size_row ; ++i) {
        for (int j=0 ; j<size_col ; ++j) {
            printf("%lf ", matrix[i*size_col+j]) ;
        }
        printf("\n");
    }
}

void print_vector (double* vector, int size) {
    for (int i=0 ; i<size ; ++i) {
        printf("%lf ", vector[i]) ;
    }
    printf("\n") ;
}

void write_vector_to_file (double* vector, int size, const char* file_path) {
    FILE* file = fopen(file_path, "w");
    if (file == NULL) {
        fprintf(stderr, "Unable to open the file %s for writing.\n", file_path);
    } else {
        for (int i = 0; i < size; ++i) {
            fprintf(file, "%lf\n", vector[i]);
        }
        printf("Vector successfully written to '%s'.\n", file_path);
        fclose(file);
    }
}

////////////////////////////////////////////////////////////////////////////////

__global__ void multiplicationMatrixVectorKernel(double* d_matrix, double* d_vector, double* d_result, int size, int T) {

    int i_block = blockIdx.x ;
    int i_block = blockIdx.y ;
    int i_thread = threadIdx.x ;
    int j_thread = threadIdx.y ;

    __shared__ double A_[T][T] ;
    __shared__ double B_[T] ;
    if (j_thread == 0) {
        double* p_d_B = d_B + (T*i_block) ;
        B_[i_thread] = p_d_B[i_thread] ;
    }

    double* p_d_A = d_A + (size*T*i_block + T*j_block) ;
    A_[i_thread][i_thread] = p_d_A[size*i_thread+j_thread] ;
    __syncthreads() ;

    double result = 0 ;
    for (int l=0 ; l<T ; ++l) {
        result += A_[i_thread][l] * B_[l] ;
    }
    __syncthreads() ;
    atomicAdd(&d_result[i_thread], result) ;
}

__global__ void getSquaredNormKernel(double* d_vector, int nb_row, double* norm) {
    unsigned int i = blockDim.x*blockIdx.x+threadIdx.x ;
    double squared_norm = 0 ;
    for (int k=0 ; k<nb_row ; k++) {
        int index = i*nb_row+k ;
        squared_norm = d_vector[index] ;
    }
    squared_norm = pow(squared_norm, 2) ;
    atomicAdd(norm, squared_norm) ;
}

__global__ void sqrtTotalNormKernel(double* norm) {
    *norm = sqrt(*norm) ;
}

__global__ void applyNormKernel(double* d_X, double* d_Y, double* norm, double* error, int size, int nb_row) {
    unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
    if (i*nb_row < size) {
        for (int k=0 ; k<nb_row ; k++) {
           int index = i*nb_row + k ;
           d_Y[index] = d_Y[index]/(*norm) ;
           double squared_error = pow((d_X[index]-d_Y[index]), 2) ;
           atomicAdd(error, squared_error) ;
        }
    }
}

__global__ void sqrtTotalErrorKernel(double* error) {
    *error = sqrt(*error) ;
}

int main (int argc, char* argv[]) {

    FILE* input_file ;

    if (argc != 2) {
        fprintf(stderr, "Usage : %s path_to_input_file\n", argv[0]);
        return 1;
    } else {
        input_file = fopen(argv[1], "r") ;
        if (input_file == NULL) {
            fprintf(stderr, "Unable to open the file %s.\n", argv[1]);
            return 1 ;
        }
    }

    int size = set_size_from_file(input_file);
    double* X = (double*) malloc (size * sizeof(double)) ;
    double* Y = (double*) malloc (size * sizeof(double)) ;

    double* A = set_flattened_matrix_from_file(input_file, size, size) ;
    fclose(input_file) ;

    for (int i=0 ; i<size ; ++i) {
        X[i] = 1.0/size ;
    }

    int block_size = size ;
    int nb_row = 25 ;
    int T = 1 ;

    // Allocation GPU
    double* d_A ;
    double* d_X ;
    double* d_Y ;
    double* error_gpu ;
    double* norm ;
    cudaMalloc((void **) &d_A, size*size);
    cudaMalloc((void **) &d_X, size);
    cudaMalloc((void **) &d_Y, size);
    cudaMalloc((void **) &error_gpu, 1);
    cudaMalloc((void **) &norm, 1);

    // Definition of GPU variables
    dim3 ThreadsPerBlock2D(T, T);
    dim3 GridSize2D(ceil(size/T), ceil(size/T));
    dim3 ThreadsPerBlockSize(block_size/nb_elem);
    dim3 GridSizeSize(size*size/block_size);
    dim3 ThreadsPerBlockUnitary(1);
    dim3 GridSizeUnitary(1);

    // TRANSFER CPU → GPU
    cudaMemcpy(d_A, A, size*size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, size*sizeof(double), cudaMemcpyHostToDevice);

    int iteration = 0;
    double zero = 0 ;
    double error_cpu = INFINITY;
    double start_time = gettimeofday_();

    // CALCULATION LOOP
    while (error_cpu > ERROR_THRESHOLD) {

        cudaMemcpy(norm, &zero, sizeof(double), cudaMemcpyHostToDevice) ;
        cudaMemcpy(error_gpu, &zero, sizeof(double), cudaMemcpyHostToDevice) ;

        // KERNEL 1 : Matrix multiplication
        multiplicationMatrixVectorKernel<<< GridSize2D, ThreadsPerBlock2D>>>(d_A, d_X, d_Y, size, T) ;

        // KERNEL 2 : Total euclidean norm
        getSquaredNormKernel<<< GridSizeSize, ThreadsPerBlockSize>>>(d_Y, nb_row, norm) ;
        sqrtTotalNormKernell<<< ThreadsPerBlockUnitary, GridSizeUnitary>>>(norm) ;

        // KERNEL 3 : Applies norm + Squared deviation
        applyNormKernel<<< GridSizeSize, ThreadsPerBlockSize>>>(d_X, d_Y, norm, error_gpu, size, nb_row) ;

        // KERNEL 4 : Total euclidean error
        sqrtTotalErrorKernel<<< ThreadsPerBlockUnitary, GridSizeUnitary>>>(error_gpu);

        //Synchronize GPU
        cudaDeviceSynchronize();

        // COMMUNICATION GPU → CPU
        cudaMemcpy(&error_cpu, error_gpu, sizeof(double), cudaMemcpyDeviceToHost);

        iteration++ ;
        printf("Iteration %d, error : %lf\n", iteration, error_cpu) ;

        // Old solution :
        // cudaMemcpy(X, d_Y, size*sizeof(double), cudaMemcpyDeviceToHost);
        // cudaMemcpy(d_X, X, size*sizeof(double), cudaMemcpyHostToDevice);
        // New solution :
        double* X_ = X;
        cudaMemcpy(X, d_Y, size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_X, X_, size*sizeof(double), cudaMemcpyHostToDevice);

    }

    // Old solution :
    // cudaMemcpy(Y, d_Y, size*sizeof(double), cudaMemcpyDeviceToHost);
    // New solution :
    // -

    double total_time = gettimeofday_() - start_time;
    printf("Time of computing : %lf\n", total_time) ;
    printf("Total number iteration : %d\n", iteration) ;
    //printf("Result : ") ;
    //print_vector(X, size) ;
    write_vector_to_file(X, size, "output.txt") ;

    // Memory release GPU and CPU
    cudaFree(d_A);
    cudaFree(d_X);
    cudaFree(d_Y);
    cudaFree(error_gpu) ;
    cudaFree(norm) ;
    free_flattened_matrix(A);
    free(X);
    free(Y);
}
