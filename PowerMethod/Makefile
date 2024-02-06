all : power_method power_method_mpi power_method_cuda_v1 power_method_cuda_v2 power_method_cuda_v3

power_method : power_method.c
		gcc -o power_method power_method.c -lm

power_method_mpi : power_method_mpi.c
		mpicc -o power_method_mpi power_method_mpi.c -lm
# RUN MPI example : mpirun -np 2 ./power_method_mpi Input/input_test_6.txt

power_method_cuda_v1 : power_method_cuda_v1.cu
		nvcc -o power_method_cuda_v1 power_method_cuda_v1.cu -lm

power_method_cuda_v2 : power_method_cuda_v2.cu
		nvcc -o power_method_cuda_v2 power_method_cuda_v2.cu -lm

power_method_cuda_v3 : power_method_cuda_v3.cu
		nvcc -o power_method_cuda_v3 power_method_cuda_v3.cu -lm

clean :
		rm *.o power_method power_method_mpi power_method_cuda_v1 power_method_cuda_v2 power_method_cuda_v3 output.txt
