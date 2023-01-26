# CSR-Matrix-Vector-Multiplication-with-MPI

Following code files perform parallelized matrix-vector multiplication of an upper triangular matrix with only odd indexed rows filled, generated in CSR form. Matrix sizes are taken as an input. The wall clock times are saved into their specified result files within the same directory.

The comparison of using sequential multiplication, multiplication with point-to-point communication and multiplication with collective communication can be made by operating on same matrix sizes.
