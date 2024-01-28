// power_method.case
// Compiler : gcc -o power_method power_method.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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

int main () {

    int size ;
    set_size(&size) ;

    double** A = (double**) malloc (size * sizeof(double*)) ;
    for (int i=0 ; i<size ; ++i) {
        A[i] = (double*) malloc (size * sizeof(double)) ;
    }
    set_matrix(A, size) ;

    printf("Matrix :\n") ;
    print_matrix(A, size) ;

    double* X = (double*) malloc (size * sizeof(double)) ;
    double* Y = (double*) malloc (size * sizeof(double)) ;

    int n = size*size ;
    for (int i=0 ; i<size ; ++i) {
        X[i] = 1.0/n ;
    }

    int iteration = 0 ;
    double difference = INFINITY ;
    while (difference > 1e-5) {
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
        // Compute difference between vector x and vector y
        difference = 0.0 ;
        for (int i=0 ; i<size ; ++i) {
            difference = pow((X[i]-Y[i]),2) ;
        }
        difference = sqrt(difference) ;
        // Increment number of iteration
        iteration += 1 ;
        // Print result
        printf("Iteration %d, difference : %lf\n", iteration, difference) ;
        // Preparing next step : Y become X
        double* X_ = X ;
        X = Y ;
        Y = X_ ;

    }

    printf("Total number iteration : %d\n", iteration) ;
    printf("Result : ") ;
    print_vector(X, size) ;

    free(X) ;
    free(Y) ;
    for (int i=0 ; i<size ; ++i) {
        free(A[i]) ;
    }
    free(A) ;

    return 0 ;
}
