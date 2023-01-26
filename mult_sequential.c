/*		
		Muhammed Enis Şen

		The following code calculates the wall clock time of an "upper triangular CSR matrix
		with odd indexed-rows filled" multiplied with a vector with sizes of taken inputs.
		It performs its operations on a single core. The calculated times will be printed
		into 'result_times_serial.txt' file within the same directory.

		To compile and run on a range of matrix sizes use the following commands:

		mpicc mult_sequential.c
		mpirun -np 1 ./a.out 10000 12500 15000
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//#include <mpi.h>
#include <unistd.h>
#include <sys/time.h>

// Random number generating function
double randBtw(double lb, double ub){
	double range = ub - lb;
	double div = RAND_MAX / range;
	return lb + (rand()/div);
}

// Fill vectors with given sizes
void fillVecRand(double *vec, int size){
	int i;
	for(i=0 ; i<size ; i++)
		vec[i] = randBtw(0., 1000.);
}

// Generate row_start and col_idx with the respective matrix size in mind
void fillCSR(int *row_start, int *col_idx, int nnz_count, int size){
	int i, j, nnz = 0;
	row_start[0] = 0;
	for(i=0 ; i<size ; i++){
		if(i%2 == 1){
			for(j=i ; j<size ; j++, nnz++)
				col_idx[nnz] = j;
			row_start[i+1] = row_start[i] + size - i;
		}else{
			row_start[i+1] = row_start[i];
		}
	}
}

// Allocate memory for Double or Int arrays
void memoryAllocationDouble(double **array, int size){
	*array = malloc(size * sizeof(double));
}
void memoryAllocationInt(int **array, int size){
	*array = malloc(size * sizeof(int));
}

// Matrix-Vector multiplication function for single-core use
void matVecMult(double *values, int *row_start, int *col_idx, double *vec, double *res_vec, int size){
	int i, j;
	double sum;
	for(i=0 ; i<size ; i+=1){
		sum = 0;
		for(j=row_start[i] ; j<row_start[i+1] ; j++)
			sum += values[j] * vec[col_idx[j]];
		res_vec[i] = sum;
		//printf("%lf - %d\n",sum, i);
	}
	printf("\n");
}

// Print row_start, col_idx and res_vec to check integrity
void printArrays(double *res_vec, int *row_start, int *col_idx, int nnz_count, int mat_size){
	int i;
	printf("\nCSR Matrix of size %dx%d with %d nnz elements\n", mat_size, mat_size, nnz_count);
	printf("\nRow_Start\n");
	for(i=0;i<mat_size+1;i++)
		printf("%d ", row_start[i]);
	printf("\nCol_Idx\n");
	for(i=0;i<nnz_count;i++)
		printf("%d ", col_idx[i]);
	printf("\nRes_Vec\n");
	for(i=0;i<mat_size;i++)
		printf("%4.6f ", res_vec[i]);
	printf("\n\n");
}

int main(int argc, char *argv[]){

	int rank, core_count, i, m, mat_size, nnz_count,
		master_rows, worker_rows, core_rows, r, elements_to_be_sent,
		total_sent, remaining_to_be_received;
	// File
	FILE *fptr;

	srand(time(NULL));

	// Take matrix sizes as inputs from run command 
	int mat_n[argc-1];
	for(i=1;i<argc;i++)
		mat_n[i-1] = atoi(argv[i]);

	for(m=0;m<argc-1;m++){

		mat_size = mat_n[m];
		nnz_count = mat_size * mat_size / 4;

		double *vec, *res_vec;
		double *values;
		int *row_start, *col_idx;

		// Allocate memory for vec and result_vec
		memoryAllocationDouble(&vec, mat_size);
		memoryAllocationDouble(&res_vec, mat_size);

		// Fill the vector to be multiplied
		fillVecRand(vec, mat_size);
		
		// Allocate memory and fill values array
		memoryAllocationDouble(&values, nnz_count);
		fillVecRand(values, nnz_count);

		// Allocate memory and fill row_start and col_idx arrays
		memoryAllocationInt(&row_start, mat_size+1);
		memoryAllocationInt(&col_idx, nnz_count);
		fillCSR(row_start, col_idx, nnz_count, mat_size);

		//printArrays(res_vec, row_start, col_idx, nnz_count, mat_size);

		struct timeval t;
		double time1, time2;
		gettimeofday(&t, NULL);
		time1 = t.tv_sec + 1.0e-6*t.tv_usec;

		// Calculate the multiplication result
		matVecMult(values, row_start, col_idx, vec, res_vec, mat_size);

		gettimeofday(&t, NULL);
		time2 = t.tv_sec + 1.0e-6*t.tv_usec;

		// Open file
		fptr = fopen("result_times_serial.txt", "a+");
		// Print resulting time in the according txt file
		fprintf(fptr, "%d %d %lf\n", 1, mat_size, time2-time1);
		printf("%d %d %lf\n", 1, mat_size, time2-time1);
		// Close file
		fclose(fptr);
		
		// Free up the allocated memory
		free(values);
		free(row_start);
		free(col_idx);
		free(res_vec);
		free(vec);
	}
	
	return 0;
}
