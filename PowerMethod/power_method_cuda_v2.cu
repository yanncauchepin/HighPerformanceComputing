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
// From v1
////////////////////////////////////////////////////////////////////////////////


__global__ void multiplicationMatrixVectorKernel(double* d_matrix, double* d_vector, double* d_result, int size) {
    unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
    if (i < size) {
        d_result[i] = 0;
        for(int j=0; j<size ; j++) {
            d_result[i] += d_matrix[i * size + j] * d_vector[j];
        }
    }
}

__global__ void getTotalNormKernel(double* d_vector, int size, double* norm) {
    *norm = 0;
    for (int i=0 ; i<size ; i++) {
        *norm += pow(d_vector[i],2);
    }
    *norm = sqrt(*norm);
}

__global__ void applyNormKernel(double* d_vector, double* norm, int size) {
    unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
    if (i < size) {
        d_vector[i] = d_vector[i]/(*norm);
    }
}

__global__ void getErrorKernel(double* d_X, double* d_Y, double* error, int size) {
    *error=0;
    for (int i=0; i<size ; i++) {
        *error += pow((d_X[i]-d_Y[i]),2);
      }
    *error = sqrt(*error);
}

////////////////////////////////////////////////////////////////////////////////
// Transform OK
////////////////////////////////////////////////////////////////////////////////


__global__ void multiplicationMatrixVectorKernel(double* d_matrix, double* d_vector, double* d_result, int size, int nb_elem, double* norm) {
    unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
    if (i*nb_elem < size) {
        for (int k=0 ; k<nb_elem ; k++) {
            int index = i*nb_elem+k ;
            d_result[index] = 0 ;
            for (int j=0 ; j<size ; j++) {
                d_result[index] += d_matrix[index*size+k] * d_vector[l] ;
            }
            double squared_result = pow(d_result[index], 2) ;
            atomicAdd(norm, squared_result) ;
        }
    }
}

__global__ void sqrtTotalNormKernel(double* norm) {
    *norm = sqrt(*norm) ;
}

////////////////////////////////////////////////////////////////////////////////
// To transform
////////////////////////////////////////////////////////////////////////////////

__global__ void applyNormKernel(double* d_vector, double* norm, int size) {
    unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
    if (i < size) {
        d_vector[i] = d_vector[i]/(*norm);
    }
}


__global__ void normKernel(REAL_T* d_Y,REAL_T* d_X, REAL_T *norm, int n, REAL_T* erreur) {
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;

  if (i*NB_ELEM < n)
  {
    REAL_T temp;
    REAL_T inter;
    for(int line = 0; line<NB_ELEM; line++)
    {
      int line_n = i*NB_ELEM+line;
      temp = d_Y[line_n]/(*norm);
      d_Y[line_n] = temp;
      inter =  d_X[line_n] - temp;
      inter = inter*inter;
      atomicAdd(erreur,inter);
    }
  }
}

__global__ void errorKernel(REAL_T *erreur) {
  *erreur = sqrt(*erreur);
}

////////////////////////////////////////////////////////////////////////////////

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

    int n = size*size ;
    for (int i=0 ; i<size ; ++i) {
        X[i] = 1.0/n ;
    }

    int block_size = size ;

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
    dim3 ThreadsPerBlockSize(block_size);
    dim3 GridSizeSize(block_size);
    dim3 ThreadsPerBlockUnitary(1);
    dim3 GridSizeUnitary(1);

    // TRANSFER CPU → GPU
    cudaMemcpy(d_A, A, size*size*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, size*sizeof(double), cudaMemcpyHostToDevice);

    int iteration = 0;
    double error_cpu = INFINITY;
    double start_time = gettimeofday_();

    // CALCULATION LOOP
    while (error_cpu > ERROR_THRESHOLD) {

        // KERNEL 1 : Matrix multiplication
        multiplicationMatrixVectorKernel<<< GridSizeSize, ThreadsPerBlockSize>>>(d_A, d_X, d_Y, size);

        // KERNEL 2 : Total euclidean norm
        getTotalNormKernel<<< ThreadsPerBlockUnitary, GridSizeUnitary>>>(d_Y, size,norm);

        // KERNEL 3 : Applies norm
        applyNormKernel<<< GridSizeSize, ThreadsPerBlockSize>>>(d_Y, norm, size);

        // KERNEL 4 : Squared deviation to get error
        getErrorKernel<<< ThreadsPerBlockUnitary, GridSizeUnitary>>>(d_Y, d_X, error_gpu, size);

        //Synchronize GPU
        cudaDeviceSynchronize();

        // COMMUNICATION GPU → CPU
        cudaMemcpy(&error_cpu, error_gpu, sizeof(double), cudaMemcpyDeviceToHost);

        iteration++ ;
        printf("Iteration %d, error : %lf\n", iteration, error_cpu) ;

        cudaMemcpy(X, d_Y, size*sizeof(double), cudaMemcpyDeviceToHost);
        cudaMemcpy(d_X, X, size*sizeof(double), cudaMemcpyHostToDevice);
    }

    cudaMemcpy(Y, d_Y, size*sizeof(double), cudaMemcpyDeviceToHost);

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
