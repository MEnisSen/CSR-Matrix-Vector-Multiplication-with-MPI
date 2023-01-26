/*		
		Muhammed Enis Şen

		The following code calculates the wall clock time of an "upper triangular CSR matrix
		with odd indexed-rows filled" multiplied with a vector with sizes of taken inputs.
		It performs its operations on collective communication. The calculated times
		will be printed into 'result_times_parallel_cc.txt' file within the same directory.

		To compile and run on a range of matrix sizes use the following commands:

		mpicc mult_collective_communication.c
		mpirun -np 4 ./a.out 10000 12500 15000
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>

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
	*array = (double *)malloc(size * sizeof(double));
}
void memoryAllocationInt(int **array, int size){
	*array = malloc(size * sizeof(int));
}

// Matrix-Vector multiplication function for single-core use
void matVecMult(double *values, int *row_start, int *col_idx, double *vec, double *res_vec, int size){
	int i, j;
	double sum;
	for(i=0 ; i<size ; i++){
		sum = 0;
		for(j=row_start[i] ; j<row_start[i+1] ; j++)
			sum += values[j] * vec[col_idx[j]];
		res_vec[i] = sum;
	}
}

// Matrix-Vector multiplication function for multi-core use
void matVecMult_Calc(double *values, int *col_idx, double *vec, double *res_vec,
					 int mat_size, int row_number, int rank, int upper_half_elements){

	int i, j, comp=0;
	double sum;

	for(i=1 ; i<=row_number ; i++){
		sum = 0;
		for(j=0 ; j<for_limit(mat_size, rank, row_number, i, 1) ; j++)
			sum += values[comp+j] * vec[col_idx[comp+j]];
		res_vec[i-1] = sum;
		comp += j;
	}

	for(i=1 ; i<=row_number ; i++){
		sum = 0;
		for(j=0 ; j<for_limit(mat_size, rank, row_number, i, 0) ; j++){
			sum += values[comp+j] * vec[col_idx[comp+j]];
		}
		res_vec[i-1+row_number] = sum;
		comp += j;
	}
}
int for_limit(int mat_size, int rank, int row_number, int i, int isUpper){
	if(isUpper) return (rank != 0) ? mat_size-(rank-1)*row_number*2-(2*i-1)
								   : mat_size/2+(2*row_number-1)-2*(i-1);
	else return (rank != 0) ? (rank*row_number-(i-1))*2-1
							: mat_size/2-(2*i-1);
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

	MPI_Status status;
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &core_count);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	srand(time(NULL) + rank);

	// Take matrix sizes as inputs from run command 
	int mat_n[argc-1];
	for(i=1;i<argc;i++)
		mat_n[i-1] = atoi(argv[i]);

	for(m=0;m<argc-1;m++){

		mat_size = mat_n[m];
		nnz_count = mat_size * mat_size / 4;

		// Calculate how many rows each core will get
		worker_rows = (mat_size / 4) / (core_count - 1);
		master_rows = (mat_size / 4) - worker_rows * (core_count - 1);
		// A unique variable for each core that takes different values if master or worker
		core_rows = (rank != 0) ? worker_rows : master_rows;

		double *vec, *res_vec, *res_vec_cores;
		double *values, *values_cores;
		int *row_start, *col_idx, *col_idx_cores;
		int *send_cnt1, *send_cnt2, *send_disp1, *send_disp2;
		double time_start, time_stop;

		// Allocate memory for vec, result_vec and res_vec_cores
		memoryAllocationDouble(&vec, mat_size);
		memoryAllocationDouble(&res_vec, mat_size/2);
		memoryAllocationDouble(&res_vec_cores, 2*core_rows);

		if (rank == 0){
			// Fill the vector to be multiplied in master core
			fillVecRand(vec, mat_size);
			
			// Allocate memory and fill values array
			memoryAllocationDouble(&values, nnz_count);
			fillVecRand(values, nnz_count);

			// Allocate memory and fill row_start and col_idx arrays
			memoryAllocationInt(&row_start, mat_size+1);
			memoryAllocationInt(&col_idx, nnz_count);
			fillCSR(row_start, col_idx, nnz_count, mat_size);

			//matVecMult(values, row_start, col_idx, vec, res_vec, mat_size);
			//printArrays(res_vec, row_start, col_idx, nnz_count, mat_size);
		}
		// Broadcast vec to every core
		MPI_Bcast(vec, mat_size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		// SCATTERV PREPARATION ----------------------------------------------
		memoryAllocationInt(&send_cnt1, core_count);
		memoryAllocationInt(&send_cnt2, core_count);
		memoryAllocationInt(&send_disp1, core_count);
		memoryAllocationInt(&send_disp2, core_count);
		elements_to_be_sent=0; total_sent=0; remaining_to_be_received=0;
		for(i=1; i<core_count; i++){
			elements_to_be_sent = mat_size * worker_rows - worker_rows * worker_rows * (2*i-1);
			remaining_to_be_received += mat_size * worker_rows - elements_to_be_sent;
			send_cnt1[i] = elements_to_be_sent;
			send_cnt2[i] = worker_rows * mat_size - elements_to_be_sent;
			total_sent += elements_to_be_sent;
			send_disp1[i] = total_sent;
			send_disp2[i] = nnz_count-remaining_to_be_received;
		}
		send_cnt1[0] = (3*nnz_count/4-total_sent);
		send_cnt2[0] = master_rows*mat_size - send_cnt1[0];
		send_disp1[0] = 0;
		send_disp2[0] = 3*nnz_count/4;
		
		int temp = send_disp1[1], temp2, last = send_disp1[core_count-1];
		send_disp1[1] = send_disp1[0];
		for(i=1;i<core_count-1;i++){
			temp2 = send_disp1[i+1];
			send_disp1[i+1] = temp;
			temp = temp2;
		}
		send_disp1[0] = last;
		// -------------------------------------------------------------------

		// Allocate memory in each core for their respective part of the matrix
		memoryAllocationDouble(&values_cores, core_rows*mat_size);
		memoryAllocationInt(&col_idx_cores, core_rows*mat_size);

		time_start = MPI_Wtime();

		// Master core scatters the values and col_idx arrays using the arrays
		// sent_cnt1, send_cnt2, send_disp1, send_disp2
		MPI_Scatterv(values, send_cnt1, send_disp1, MPI_DOUBLE,
					 values_cores, send_cnt1[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Scatterv(values, send_cnt2, send_disp2, MPI_DOUBLE,
					 &values_cores[send_cnt1[rank]], send_cnt2[rank], MPI_DOUBLE, 0, MPI_COMM_WORLD);

		MPI_Scatterv(col_idx, send_cnt1, send_disp1, MPI_INT,
					 col_idx_cores, send_cnt1[rank], MPI_INT, 0, MPI_COMM_WORLD);
		MPI_Scatterv(col_idx, send_cnt2, send_disp2, MPI_INT,
					 &col_idx_cores[send_cnt1[rank]], send_cnt2[rank], MPI_INT, 0, MPI_COMM_WORLD);

		// Master calculates its own part if any rows are left after the scatter step
		if (core_rows != 0){
			matVecMult_Calc(values_cores, col_idx_cores, vec, res_vec_cores, mat_size, core_rows, rank, elements_to_be_sent);
		}

		// GATHERV PREPARATION -----------------------------------------------
		total_sent=0; remaining_to_be_received=mat_size/4;
		send_cnt1[0] = master_rows;
		send_disp1[0] = mat_size/4-master_rows;
		send_disp2[0] = mat_size/4;
		for(i=1; i<core_count; i++){
			send_cnt1[i] = worker_rows;
			
			send_disp1[i] = (i == 1) ? 0 : send_disp1[i-1] + worker_rows;
			send_disp2[i] = mat_size/4+master_rows+(core_count-i-1)*worker_rows;
		}
		// -------------------------------------------------------------------

		// Master core gathers the values_cores and col_idx_cores arrays using the arrays
		// sent_cnt1, send_disp1, send_disp2
		MPI_Gatherv(res_vec_cores, core_rows, MPI_DOUBLE,
					res_vec, send_cnt1, send_disp1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		MPI_Gatherv(res_vec_cores, core_rows, MPI_DOUBLE,
					res_vec, send_cnt1, send_disp2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		time_stop = MPI_Wtime();

		if (rank == 0){
			// Open file
			fptr = fopen("result_times_parallel_cc.txt", "a+");
			// Print resulting time in the according txt file
			fprintf(fptr, "%d %d %lf\n", core_count, mat_size, time_stop-time_start);
			//printf("%d %d %lf\n", core_count, mat_size, time_stop-time_start);
			// Close file
			fclose(fptr);
		}

		// Free up the allocated memory
		if(rank == 0){
			/*for(i=0;i<mat_size/2;i++)
				printf("%lf\n", res_vec[i]);*/
			free(values);
			free(row_start);
			free(col_idx);
		}
		free(values_cores);
		free(col_idx_cores);
		free(res_vec);
		free(res_vec_cores);
		free(vec);
		free(send_cnt1);
		free(send_cnt2);
		free(send_disp1);
		free(send_disp2);
	}

	MPI_Finalize();
	
	return 0;
}
