// cholesky.cpp

#include <iostream>
#include <cmath>

double** transpose (double** matrix, int size_row, int size_col) {
    double** transpose = new double*[size_col] ;
    for (int j = 0; j < size_col; ++j) {
        transpose[j] = new double[size_row] ;
    }
    for (int i = 0; i < size_row; ++i) {
        for (int j = 0; j < size_col; ++j) {
            transpose[j][i] = matrix[i][j];
        }
    }
    return transpose ;
}

double* flogit (double* y, int size, double k) {
    double* logit = new double[size] ;
    for (int i = 0; i < size; ++i) {
        logit[i] = log((y[i] / k) / (1 - (y[i] / k)));
    }
    return logit ;
}

double* multiply_matrix_by_vector (double** matrix, double* vector, int size_row, int size_col) {
    double* result = new double[size_row] ;
    for (int i = 0; i < size_row; ++i) {
        for (int j = 0; j < size_col; ++j) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    return result ;
}

double** multiply_transposed_matrix_by_matrix(double** A, int size_row, int size_col) {
    double** AT = transpose(A, size_row, size_col) ;
    double** ATA = new double*[size_col] ;
    for (int i = 0; i<size_col ; ++i) {
        ATA[i] = new double[size_col] ;
    }
    for (int i = 0; i < size_col; ++i) {
        for (int j = 0; j < size_col; ++j) {
            ATA[i][j] = 0.0 ;
            for (int k = 0; k < size_row; ++k) {
                ATA[i][j] += AT[i][k] * A[k][j];
            }
        }
    }
    return ATA ;
}

double* forward_substitution (double** L, double* w, int size) {
    // Original equation : A AT y = AT b
    // L : lower triangular matrix where AT A = L LT
    // FORWARD SUBSITUTION
    // w = AT b and y = LT xy
    // Resolve L y = w to get y
    double* y = new double[size] ;
    for (int i = 0; i < size; ++i) {
        double sum = 0;
        for (int j = 0; j < i; ++j) {
            sum += L[i][j] * y[j];
        }
        y[i] = (w[i] - sum) / L[i][i];
    }
    return y ;
}

double* backward_substitution (double** LT, double* y, int size) {
    // Original equation : A AT y = AT b
    // LT : upper triangular matrix = transpose(L) where AT A = L LT
    // FORWARD SUBSITUTION
    // Resolve LT x = y to get x
    double* x = new double[size] ;
    for (int i = size - 1; i >= 0; --i) {
        double sum = 0;
        for (int j = size - 1; j > i; --j) {
            sum += LT[i][j] * x[j];
        }
        x[i] = (y[i] - sum) / LT[i][i];
    }
    return x ;
}

double** cholesky (double** ATA, int size) {
    double** L = new double*[size] ;
    for (int i = 0; i<size ; ++i) {
        L[i] = new double[size] ;
    }
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            L[i][j] = 0.0;
        }
    }
    L[0][0] = sqrt(ATA[0][0]) ;

    // ERROR
    for (int i = 1; i < size; ++i) {
        //L[i][0] = ATA[i][0] / L[0][0];
        L[i][0] = ATA[i][0] / sqrt(ATA[0][0]) ;
    }
    /*
    for (int i = 1 ; i < size ; ++i) {
        double sum = 0 ;
        for (int k = 0 ; k < i ; ++k) {
            sum += pow(L[i][k], 2) ;
        }
        L[i][i] = sqrt(ATA[i][i] - sum) ;
        for (int p = i + 1; p < size; ++p) {
            sum = 0;
            for (int k = 0; k < i; ++k) {
                sum += L[i][k] * L[p][k];
            }
            L[p][i] = (ATA[i][p] - sum) / L[i][i];
        }
    }
    */
    return L ;
}

int main() {

    int size = 10 ;
    double** A = new double*[size] ;
    for (int i=0 ; i<size ; ++i) {
        A[i] = new double[2] ;
    }
    for (int i=0; i<size ; ++i) {
        A[i][0] = i ;
        A[i][1] = -1 ;
    }
    double** AT = transpose(A, size, 2) ;
    double* y_input = new double[size]{0.53, 0.53, 1.53, 2.53, 12.53, 21.53, 24.53, 28.53, 28.53, 30.53} ;
    int K = 30.54 ;
    double* b = flogit(y_input, size, K) ;
    double** ATA = multiply_transposed_matrix_by_matrix(A, size, 2) ;

    // ERROR
    double** L = cholesky(ATA, size) ;
    /*
    double* w = multiply_matrix_by_vector(AT, b, size, 2) ;
    double* y = forward_substitution(L, w, size) ;
    double** LT = transpose(L, size, size) ;
    double* x = backward_substitution(LT, y, size) ;
    std::cout << "Result : " << x[0] << ", " << x[1] <<std::endl ;
    */
}
