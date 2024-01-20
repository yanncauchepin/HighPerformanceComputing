      PROGRAM CALCUL
      INTEGER MMAX, NMAX
      PARAMETER (MMAX = 50)
      PARAMETER (NMAX = 10)
      DOUBLE PRECISION A(MMAX,NMAX),ATA(NMAX,NMAX),L(NMAX,NMAX)
      DOUBLE PRECISION LT(NMAX,NMAX), YY(MMAX) 
      DOUBLE PRECISION B(MMAX),X(NMAX),W(NMAX),Y(NMAX),XX(MMAX)
      DOUBLE PRECISION K
      INTEGER M, N, i
      K = 30.54
      N = 2
      WRITE (*,*) "Entrez les Xi :"
      CALL DREAD_MPL (M, 1, XX, MMAX)
      do i=1,M
         A(i,1)=XX(i)
         A(i,2)=-1
      end do
      CALL DPRINT_MPL ('Matrice A', M, N, A, MMAX)
      WRITE (*,*) "Entrez les Yi :"
      CALL DREAD_MPL (M,1,YY,MMAX)
      CALL FLOGIT (B,YY, M, MMAX, K)
      CALL DPRINT_MPL ('Matrice B', M, 1, B, MMAX)
      CALL DGEMM ('T','N',N,M,M,1D0,A,MMAX,A,MMAX,0D0,ATA,NMAX)
      CALL DPRINT_MPL ('Matrice ATA', N, N, ATA, NMAX)
      CALL CHOLESKY (L,ATA,N,NMAX)
      WRITE (*,*) "PROCEDURE DE CHOLESKY"
      CALL DPRINT_MPL ('Matrice L', N, N, L, NMAX)
      WRITE (*,*) "PROCEDURE DE DESCENTE ET REMONTE"
      CALL MULTI_MATT_VEC (A, B, N, M, MMAX, NMAX, W)
      CALL DPRINT_MPL ('Matrice W', N, 1, W, NMAX)
      CALL DESCENTE (L, N, NMAX, W, Y)
      CALL DPRINT_MPL ('Matrice Y', N, 1, Y, NMAX)
      CALL TRANSP (L, LT, N, N, NMAX, NMAX)
      CALL REMONTE (LT, N, NMAX, Y, X)
      CALL DPRINT_MPL ('Matrice X', N, 1, X, NMAX)
      END PROGRAM

      
      SUBROUTINE TRANSP (A, AT, N, M, NMAX, MMAX) 
      INTEGER i, j, N, M, NMAX, MMAX
      DOUBLE PRECISION A(MMAX,NMAX), AT(NMAX,MMAX)
      do i=1,N
         do j=1,M
             AT(i,j)=A(j,i)
         end do
      end do
      end subroutine
                
                
      SUBROUTINE FLOGIT (B,Y, M, MMAX, K)
      DOUBLE PRECISION B(MMAX), Y(MMAX), K
      INTEGER M, MMAX, i
      do i=1,M
         B(i)=LOG((Y(i)/K)/(1-(Y(i)/K)))
      end do
      end subroutine
      
      
      SUBROUTINE MULTI_MATT_VEC (A, B, N, M, MMAX, NMAX, AB)
      INTEGER i, j, MMAX, NMAX, N, M
      DOUBLE PRECISION A(MMAX,NMAX), AT(NMAX,MMAX)
      DOUBLE PRECISION B(MMAX), AB(NMAX)
      CALL TRANSP (A, AT, N, M, NMAX, MMAX)
      do i=1,N
         do j=1,M
            AB(i) = AB(i) + AT(i,j)*B(j)
         end do
      end do
      end subroutine

      
      SUBROUTINE DESCENTE(L,N,NMAX,B,X)
      INTEGER i,j,N,NMAX
      DOUBLE PRECISION  X(NMAX), B(NMAX)
      DOUBLE PRECISION L(NMAX,NMAX)
      DOUBLE PRECISION somme
      somme = 0
      do i=1,N
         do j=1,i-1
            somme = somme + L(i,j)*X(j)
         end do
         X(i)=(B(i)-somme)/L(i,i)
      end do
      end subroutine
      
      
      SUBROUTINE REMONTE(L,N,NMAX,B,X)
      INTEGER i,j,N,NMAX
      DOUBLE PRECISION x(NMAX), B(NMAX)
      DOUBLE PRECISION L(NMAX,NMAX),somme
      somme = 0
      do i=N,1,-1
         do j=N,i+1,-1
            somme = somme + L(i,j)*X(j)
         end do
         X(i)=(B(i)-somme)/L(i,i)
      end do
      end subroutine


      SUBROUTINE CHOLESKY (L,ATA,N,NMAX)
      INTEGER NMAX
      INTEGER n,i,p,k
      DOUBLE PRECISION somme, L(NMAX,NMAX), ATA(NMAX,NMAX)   
      L(1,1) = sqrt(ATA(1,1))
      do i=2,N                 
         L(i,1)=ATA(i,1)/L(1,1)
      end do
      do i=2,N                 
         somme = 0
         do k=1,i-1           
            somme=somme+(L(i,k))**2
         end do
         L(i,i) = sqrt(ATA(i,i) - somme)
         do p=i+1,n
         somme = 0
            do k=1,i-1
               somme=somme+L(i,k)*L(p,k)
            end do
            L(p,i) = (ATA(i,p)-somme)/L(i,i)
         end do
      end do
      end subroutine
