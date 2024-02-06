// power_method_mpi.c
// Compile : mpicc -o power_method_mpi power_method_mpi.c -lm
// Run : mpirun -np 2 ./power_method_mpi input_test_6.txt

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>
#include <mpi.h>

#define ERROR_THRESHOLD 1e-5

/*
TO DO LIST :
- When divide block_size, handle case size/number_processes = int + decimal where decimal non null
- Check Optimize reading set_flattened_submatrix_from_file : error in pointer fseek depending on decimal of double values for example.
- Check optimization with Ssend for example
*/

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

double* set_flattened_submatrix_from_file (FILE* file, int start_row, int size_row, int size_col) {
    rewind(file) ;
    double* matrix = build_flattened_matrix (size_row, size_col) ;


    int size = set_size_from_file (file) ;
    double value ;
    for (int i=0 ; i < start_row ; i++) {
        for (int j=0 ; j < size_col ; j++) {
            if (fscanf(file, "%lf ", &value) != 1) {
                fprintf(stderr, "Error reading the element [%d][%d] of the matrix in the file.\n", i + start_row, j);
                fclose(file);
                free_flattened_matrix(matrix);
                exit(EXIT_FAILURE);
            }
        }
    }
    // ALTERNATIVE WITH ERRORS : seek(file, sizeof(int) + (start_row*size_col)*sizeof(double), SEEK_SET);


    for (int i = 0 ; i < size_row ; i++) {
        for (int j = 0; j < size_col ; j++) {
            if (fscanf(file, "%lf ", &matrix[i*size_col+j]) != 1) {
                fprintf(stderr, "Error reading the element [%d][%d] of the matrix in the file.\n", i + start_row, j);
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

    // MPI parameters
    int number_processes ;
    int rank ;
    MPI_Init(&argc,&argv) ;
    MPI_Status status ;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank) ;
    MPI_Comm_size(MPI_COMM_WORLD,&number_processes) ;
  	int block_size = size / number_processes ;

  	//printf("PROCESS %d\n", rank);

    double pseudo_norm ;
  	double total_norm ;
  	double pseudo_error ;
  	double total_error ;
  	int finish = 0 ;

    if (rank != 0) {
        /*
        double* block = build_flattened_matrix(block_size, size) ;
        MPI_Recv (block, block_size*size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &status);
        printf("PROCESS %d | RECEIVE block matrix\n", rank);
        print_flattened_matrix(block, block_size, size) ;
        while (finish == 0) {
            MPI_Bcast (X, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            printf("PROCESS %d | RECEIVE X\n", rank);
            pseudo_norm = 0;
            for (int i=0 ; i<block_size ; i++) {
                Y[i] = 0;
                for (int j=0 ; j<size ; j++) {
                    Y[i] += block[i*size+j] * X[j];
                }
                pseudo_norm += Y[i]*Y[i];
          }
          MPI_Reduce (&pseudo_norm, &total_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          printf("PROCESS %d | SEND pseudo_norm\n", rank);
          MPI_Bcast (&total_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          printf("PROCESS %d | RECEIVE total_norm\n", rank);
          pseudo_error = 0;
          for (int i=0 ; i<block_size ; i++) {
              Y[i] = Y[i]/total_norm;
              pseudo_error += pow((X[i]-Y[i]),2);
          }
          MPI_Reduce(&pseudo_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
          printf("PROCESS %d | SEND pseudo_error\n", rank);
          MPI_Gather(Y, block_size, MPI_DOUBLE, X, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          printf("PROCESS %d | SEND Y\n", rank);
          MPI_Bcast(&finish, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
          printf("PROCESS %d | RECEIVE finish\n", rank);
        }
        free_flattened_matrix(block);
        */
    } else {
        ////////////////////////////////////////////////////////////////////////
        double* block_test ;
        block_test = set_flattened_submatrix_from_file(input_file, 0, block_size, size) ;
        printf("\nstart 0 :\n") ;
        print_flattened_matrix(block_test, block_size, size) ;
        block_test = set_flattened_submatrix_from_file(input_file, 1, block_size, size) ;
        printf("\nstart 1 :\n") ;
        print_flattened_matrix(block_test, block_size, size) ;
        block_test = set_flattened_submatrix_from_file(input_file, 2, block_size, size) ;
        printf("\nstart 2 :\n") ;
        print_flattened_matrix(block_test, block_size, size) ;
        block_test = set_flattened_submatrix_from_file(input_file, block_size, block_size, size) ;
        printf("\nstart %d :\n", block_size) ;
        print_flattened_matrix(block_test, block_size, size) ;
        free_flattened_matrix(block_test) ;
        ////////////////////////////////////////////////////////////////////////
        /*
        double* block = build_flattened_matrix(block_size, size) ;
        for (int i=1 ; i<number_processes ; i++) {
      			block = set_flattened_submatrix_from_file (input_file, i*block_size, block_size, size) ;
      			MPI_Send (block, block_size*size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
            printf("PROCESS %d | SEND block matrix\n", rank) ;
            print_flattened_matrix(block, block_size, size) ;
    		}
        block = set_flattened_submatrix_from_file (input_file, 0, block_size, size) ;
        for (int i=0 ; i<size ; ++i) {
            X[i] = 1.0/size ;
        }
    		int iteration = 0;
    		total_error = INFINITY;
        double start_time = gettimeofday_();
    		while (finish == 0) {
            MPI_Bcast (X, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            printf("PROCESS %d | SEND X\n", rank) ;
      			pseudo_norm = 0;
      			for (int i=0 ; i<block_size ; i++) {
                Y[i] = 0;
        				for(int j=0 ; j<size ; j++) {
                    Y[i] += block[i*size+j] * X[j];
        				}
        				pseudo_norm += Y[i]*Y[i];
      			}
      			MPI_Reduce (&pseudo_norm, &total_norm, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            printf("PROCESS %d | RECEIVE pseudo_norm\n", rank);
            printf("PROCESS %d | COMPUTE total_norm\n", rank);
      			total_norm = sqrt(total_norm);
      			MPI_Bcast (&total_norm, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            printf("PROCESS %d | SEND total_norm\n", rank) ;
      			pseudo_error = 0;
      			for (int i=0 ; i<block_size ; i++) {
        		    Y[i] = Y[i]/total_norm;
        				pseudo_error += pow((X[i] - Y[i]),2);
      			}
      			MPI_Reduce (&pseudo_error, &total_error, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
            printf("PROCESS %d | RECEIVE pseudo_error\n", rank);
            printf("PROCESS %d | COMPUTE total_error\n", rank);
      			MPI_Gather (Y, block_size, MPI_DOUBLE, X, block_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            printf("PROCESS %d | RECEIVE Y\n", rank);
      			if (total_error > ERROR_THRESHOLD) {
                finish = 0;
      			} else {
                finish = 1;
      			}
            MPI_Bcast(&finish, size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
            printf("PROCESS %d | SEND finish\n", rank);
      			iteration++;
            printf("Iteration %d, error : %lf\n", iteration, total_error) ;
    		}
        double total_time = gettimeofday_() - start_time;
        printf("Time of computing : %lf\n", total_time) ;
        printf("Total number iteration : %d\n", iteration) ;
        write_vector_to_file(X, size, "output.txt") ;
        free_flattened_matrix(block);
        */
    }
    free(Y);
    free(X);
    MPI_Finalize();
    return 0;
}
