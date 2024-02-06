// power_method.c
// Compile : gcc -o power_method power_method.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#define ERROR_THRESHOLD 1e-5

double gettimeofday_ () {
    struct timeval time ;
    gettimeofday (&time, NULL) ;
    return time.tv_sec + (time.tv_usec * 1.0e-6L);
}

double** build_matrix (int size) {
    double** matrix = (double**) malloc (size * sizeof(double*)) ;
    for (int i=0 ; i<size ; ++i) {
        matrix[i] = (double*) malloc (size * sizeof(double)) ;
    }
    return matrix ;
}

void free_matrix (double** matrix, int size) {
    for (int i=0 ; i<size ; ++i) {
        free(matrix[i]) ;
    }
    free(matrix) ;
}

void set_size (int* size) {
    printf("Enter the size of the squarred matrix : ") ;
    scanf("%d", size) ;
}

void set_matrix (double** matrix, int size) {
    for (int i=0 ; i<size ; ++i) {
        for (int j=0 ; j<size ; ++j) {
            printf("Enter the value of matrix A[%d][%d] : ", i, j) ;
            scanf("%lf", &matrix[i][j]) ;
        }
    }
}

double** set_matrix_from_file (FILE* file, int* size) {
    if (fscanf(file, "%d", size) != 1) {
        fprintf(stderr, "Error reading the size of the matrix in the file.\n");
        fclose(file);
        exit(EXIT_FAILURE);
    }
    double** matrix = build_matrix(*size) ;
    for (int i = 0; i < *size; i++) {
        for (int j = 0; j < *size; j++) {
            if (fscanf(file, "%lf", &matrix[i][j]) != 1) {
                fprintf(stderr, "Error reading the element [%d][%d] of the matrix in the file.\n", i, j);
                fclose(file);
                free_matrix(matrix, *size) ;
                exit(EXIT_FAILURE);
            }
        }
    }
    return matrix ;
}

void print_matrix (double** matrix, int size) {
    for (int i=0 ; i<size ; ++i) {
        for (int j=0 ; j<size ; ++j) {
            printf("%lf ", matrix[i][j]) ;
        }
        printf("\n");
    }
}

void print_vector (double* vector,int size) {
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

int main (int argc, char* argv[]) {

    int size ;
    double** A ;

    if (argc == 1) {
        set_size(&size) ;
        A = build_matrix(size) ;
        set_matrix(A, size) ;
    } else if (argc == 2) {
        FILE* input_file ;
        input_file = fopen(argv[1], "r") ;
        if (input_file == NULL) {
            fprintf(stderr, "Unable to open the file %s.\n", argv[1]);
            return 1 ;
        }
        A = set_matrix_from_file(input_file, &size) ;
        fclose(input_file) ;
    } else {
        fprintf(stderr, "Usage 1 : %s path_to_input_file\nUsage 2 : %s \n", argv[0],  argv[0]);
        return 1;
    }


    //printf("Matrix :\n") ;
    //print_matrix(A, size) ;

    double* X = (double*) malloc (size * sizeof(double)) ;
    double* Y = (double*) malloc (size * sizeof(double)) ;

    for (int i=0 ; i<size ; ++i) {
        X[i] = 1.0/size ;
    }

    int iteration = 0 ;
    double error = INFINITY ;
    double start_time = gettimeofday_() ;
    while (error > ERROR_THRESHOLD) {
        // Multiply matrix A by vector x to get vector y
        for (int i=0 ; i<size ; ++i) {
            Y[i] = 0.0 ;
            for (int j=0 ; j<size ; ++j) {
                Y[i] += A[i][j] * X[j] ;
            }
        }
        // Normalization of vector y
        double sum = 0.0 ;
        for (int i=0 ; i<size ; ++i) {
            sum += pow(Y[i],2) ;
        }
        sum = sqrt(sum) ;
        for (int i=0 ; i<size ; ++i) {
            Y[i] = Y[i]/sum ;
        }
        // Compute error between vector x and vector y
        error = 0.0 ;
        for (int i=0 ; i<size ; ++i) {
            error = pow((X[i]-Y[i]),2) ;
        }
        error = sqrt(error) ;
        // Increment number of iteration
        iteration += 1 ;
        // Print result
        printf("Iteration %d, error : %lf\n", iteration, error) ;
        // Preparing next step : Y become X
        double* X_ = X ;
        X = Y ;
        Y = X_ ;

    }

    double total_time = gettimeofday_() - start_time ;
    printf("Time of computing : %lf\n", total_time) ;
    printf("Total number iteration : %d\n", iteration) ;
    //printf("Result : ") ;
    //print_vector(X, size) ;
    write_vector_to_file(X, size, "output.txt") ;

    free_matrix(A, size) ;
    free(X) ;
    free(Y) ;

    return 0 ;
}
