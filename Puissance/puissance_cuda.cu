//puissance_cuda.c

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "defs.h"

#define TAILLE_BLOC_X 256

//nb de bloc : gridDim.x
//indice de parcours des blocs : blockIdx.x

//nb de thread par bloc : blockDim.x
//indice de parcours des threads : threadIdx.x

__global__ void matmulKernel(float* d_A, float* d_B, float* d_C, int n) {
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i < n){
    float temp=0;
    for(int k=0; k<n; k++)
      temp = temp + d_A[i * n + k] * d_B[k];
    d_C[i] = temp;
  }
}

__global__ void norm_tot_Kernel(float* d_C, int n,double *norm) {
  *norm=0;
  for(int k=0; k<n; k++)
      *norm += d_C[k]*d_C[k];

  *norm = sqrt(norm);

}


__global__ void normKernel(float* d_Y,float* norm, int n) {
  unsigned int i = blockDim.x*blockIdx.x+threadIdx.x;
  if (i < n){
    d_Y[i] = d_Y[i]/(*norm);
  }
}

__global__ void errorKernel(float* d_X, float*d_Y,double *erreur) {
  *erreur=0;
  for(int k=0; k<n; k++)
      *erreur += d_X[k]*d_Y[k];

  *erreur = sqrt(*erreur);

}


//Création de la matrice sur CPU pour accès GPU
int main(int argc, char **argv){
    long i, j, k, n;
    long long size;
    REAL_T norm, inv_norm, error, delta;
    REAL_T *A, *A_i, *X, *Y;
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
    error = INFINITY;
    n_iterations = 0;

    // Allocation GPU
    float *d_A, *d_X,*d_Y, *error, *norm;
    cudaMalloc((void **) &d_A, size);
    cudaMalloc((void **) &d_X, n);
    cudaMalloc((void **) &d_Y, n);
    cudaMalloc((void **) &error, 1);
    cudaMalloc((void **) &norm, 1);

    //Transfert CPU → GPU
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_X, X, n, cudaMemcpyHostToDevice);

    //BOUCLE DE CALCUL
    while (error > ERROR_THRESHOLD) {

      //KERNEL 1 : Multipication matrice
      //Lancement de kernel (asynchrone) :
      //Définition des variables GPU
      dim3 threadsParBloc(TAILLE_BLOC_X);
      dim3 tailleGrille(ceil(n/(float) TAILLE_BLOC_X));
      matmulKernel<<< tailleGrille, threadsParBloc>>>(d_A, d_X, d_Y, n);

      //KERNEL 2 : Norme euclidienne total
      //Définition des variables GPU
      dim3 threadsParBloc(1);
      dim3 tailleGrille(1);
      normtotKernel<<< threadsParBloc, tailleGrille>>>(d_Y, n,norm);

      //KERNEL 3 : Applique la norme
      dim3 threadsParBloc(TAILLE_BLOC_X);
      dim3 tailleGrille(ceil(n/(float) TAILLE_BLOC_X));
      normKernel<<< tailleGrille, threadsParBloc>>>(d_Y, norm, n);

      // KERNEL 4 : Ecart quadratique
      dim3 threadsParBloc(1);
      dim3 tailleGrille(1));
      errorKernel<<< threadsParBloc, tailleGrille>>>(d_Y, d_X, error);

      //COMMUNICATION GPU → CPU
      cudaMemcpy(error, error, 1, cudaMemcpyDeviceToHost);

      n_iterations ++ ;
    }

    cudaMemcpy(Y, d_Y, taille_totale, cudaMemcpyDeviceToHost);

    total_time = my_gettimeofday() - start_time;
    printf("erreur finale après %4d iterations : %g (|VP| = %g)\n", n_iterations, error, norm);
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
    free(A); free(X); free(Y);

  }
