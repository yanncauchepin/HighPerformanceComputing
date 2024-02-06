//puissance_cuda.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "defs.h"

#include <cuda_runtime.h>
#include "cublas_v2.h"

//nb de bloc : gridDim.x
//indice de parcours des blocs : blockIdx.x

//nb de thread par bloc : blockDim.x
//indice de parcours des threads : threadIdx.x


//Création de la matrice sur CPU pour accès GPU
int main(int argc, char **argv){

    cublasHandle_t handle;

    long i, n;
    long long size;
    REAL_T error_cpu,norm_cpu;
    error_cpu = 99;
    REAL_T *A, *A_i, *X, *Y, *d_A, *d_X,*d_Y,*d_error;
    double start_time, total_time;
    int n_iterations;
    FILE *output;



    if (argc < 2){
        printf("USAGE: %s [n]\n", argv[0]);
        exit(1);
    }
    n = atoll(argv[1]);
    size = n * n * sizeof(REAL_T);
    printf("taille de la matrice : %.1f G\n", size / 1073741824.);

    /*** allocation de la matrice et des vecteurs ***/
    A = (REAL_T *)malloc(size);
    if (A == NULL) {
        perror("impossible d'allouer la matrice");
        exit(1);
    }
    X = (REAL_T *)malloc(n * sizeof(REAL_T));
    Y = (REAL_T *)malloc(n * sizeof(REAL_T));
    if ((X == NULL) || (Y == NULL)) {
        perror("impossible d'allouer les vecteur");
        exit(1);
    }

    /*** initialisation de la matrice et de x ***/
    A_i = A;
    for (i = 0; i < n; i++) {
        init_ligne(A_i, i, n);
        A_i += n;
    }

    for (i = 0; i < n; i++) {
        X[i] = 1.0 / n;
    }

    //Initialisation de variables
    start_time = my_gettimeofday();
    n_iterations = 0;

    // Allocation GPU
    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_X, n * sizeof(REAL_T));
    cudaMalloc((void **) &d_Y, n * sizeof(REAL_T));
    cudaMalloc((void **) &d_error, n * sizeof(REAL_T));


    //Transfert CPU → GPU
    cublasSetVector(n, sizeof(float), X, 1, d_X, 1);
    cublasSetMatrix(n, n, sizeof(float), A, n, d_A, n);

    cublasCreate(&handle);



    //BOUCLE DE CALCUL
    while (error_cpu > ERROR_THRESHOLD) {

      //KERNEL 1 : Multipication matrice
      //Lancement de kernel (asynchrone) :
      //Définition des variables GPU

      float alpha = 1;
      float beta = 0;

      cublasSgemv(handle, CUBLAS_OP_T, n, n, &alpha, d_A, n, d_X, 1, &beta, d_Y, 1);

      //KERNEL 2 : Norme euclidienne total
      //Définition des variables GPU
      cublasSnrm2(handle, n, d_Y, 1, &norm_cpu);


      //KERNEL 3 : Applique la norme
      norm_cpu = 1/norm_cpu;
      cublasSscal(handle, n, &norm_cpu, d_Y, 1);

      // KERNEL 4 : Ecart quadratique
      alpha = -1;
      cublasScopy(handle, n, d_Y, 1,d_error, 1);
      cublasSaxpy(handle, n, &alpha, d_X, 1, d_error, 1);
      cublasSnrm2(handle, n, d_error, 1, &error_cpu);

      //COMMUNICATION GPU -> CPU
      cublasScopy(handle, n, d_Y, 1,d_X, 1);


      n_iterations ++ ;

    }
    cublasGetVector(n, sizeof(float), d_Y, 1, Y, 1);

    total_time = my_gettimeofday() - start_time;

    printf("erreur finale après %4d iterations : %g (|VP| = %g)\n", n_iterations, error_cpu, 1/norm_cpu);
    printf("temps : %.1f s      Mflop/s : %.1f \n", total_time, (2.0 * n * n + 7.0 * n) * n_iterations / 1048576. / total_time);
    /*** stocke le vecteur propre dans un fichier ***/
    output = fopen("result.out", "w");
    if (output == NULL) {
        perror("impossible d'ouvrir result.out en écriture");
        exit(1);
    }
    fprintf(output, "%ld\n", n);
    for (i = 0; i < n; i++) {
        fprintf(output, "%.17g\n", Y[i]);
    }
    fclose(output);

    /* Libération mémoire GPU et CPU : */

    cudaFree(d_A); cudaFree(d_X); cudaFree(d_Y);
    cublasDestroy(handle);
    free(A); free(X); free(Y);

  }
