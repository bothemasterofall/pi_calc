
/**********************************************************************
*** NAME           : Bo Cimino                                      ***
*** CLASS          : CSc 318                                        ***
*** DUE DATE       : 04/28/2014                                     ***
*** INSTRUCTUOR    : GAMRADT                                        ***
***********************************************************************
*** DESCRIPTION: 	This program calculates pi by integrating the function
					1/(1+x^2) on the interval 0->1 using a riemann sum.
					Because millions or billions of partitions are used
					in this riemann sum, parallel programming techniques
					are implemented using mpi and openmp.
					At the end, the approximate value of pi is output,
					along with the real value of pi and the time it took
					to calculate it. The number of threads per process
					is also displayed.
					Adding more processes should not speedup computation time
					as each process performs the same number of partitions.
					More threads will speed up time.
**********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <omp.h>

#define PI 3.1415926535897932384626433;

void welcome();
void ReadCommandLine(int argc, char* argv[], double* xInit, double* xEnd, unsigned long* numParts);
void calcPi(double xInit, double xEnd, unsigned long partitions, double* glob_total, int my_rank, int comm_size, int *num_threads);
double midpoint(double start, unsigned long partitions, double width, int* num_threads);
double calcFx(double x);
void display(double myPie, double xInit, double xEnd, unsigned long partitions, double time, int comm_size, int num_threads);

int main(int argc, char* argv[])
{
    int my_rank, comm_size, num_threads = 0; 
	unsigned long partitions = 0, local_partitions;
	double local_integral = 0.0, total = 0.0, start = 0.0, end = 0.0, local_start, width, process_width;
    double ts = 0.0, tf = 0.0, tr = 0.0;

    /*initilaize MPI*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if(my_rank == 0)
    {
        welcome();
        ReadCommandLine(argc, argv, &start, &end, &partitions);
        ts = MPI_Wtime();
    }

    /*broadcast data*/
    MPI_Bcast(&start,1,MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&end,1,MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&partitions,1,MPI_DOUBLE, 0, MPI_COMM_WORLD);

    /*common to all processes*/
    width = (end - start)/(partitions * comm_size);
    local_partitions = partitions;
    process_width = local_partitions * width;
	
    /*unique to each process*/
    local_start = start + (my_rank * process_width);
    local_integral = midpoint(local_start, local_partitions, width, &num_threads);

    MPI_Reduce(&local_integral, &total, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if(my_rank == 0)
    {
        tf = MPI_Wtime();
        tr = tf - ts;
        display(total, start, end, partitions, tr, comm_size, num_threads);
    }
    MPI_Finalize();
    return 0;
}

/**********************************************************************************
*** FUNCTION welcome                                                            ***
***********************************************************************************
*** DESCRIPTION : Welcomes the user and explains the program to them            ***
*** INPUT ARGS  : none                                                          ***
*** OUTPUT ARGS : none                                                          ***
*** IN/OUT ARGS : none                                                          ***
*** RETURN      : void                                                          ***
**********************************************************************************/
void welcome()
{
    printf("This program will calculate pi.\n");
	printf("The function 1/(1+x^2) is evaluated from 0 to 1 using a Riemann sum.\n");
	printf("The number of rectangles that the function is divided into is specified\nin the pbs file.\n");
	printf("Because millions (or billions) of partitions are used,\n");
	printf("parallelism is implemented using mpi and openmp to speed up the computation.\n");
	printf("Each process created by mpi spawns multiple threads. Each process has the same\n");
	printf("number of partitions, so more or less process should not speed up computation time,");
	printf("but more threads will cause linear speedup.");
}

/**********************************************************************************
*** FUNCTION ReadCommandLine                                                    ***
***********************************************************************************
*** DESCRIPTION : This function reads in input from the command line. In this program
                  this is the number of partitions desired for each process     ***
*** INPUT ARGS  : argc, argv                                                    ***
*** OUTPUT ARGS : none                                                          ***
*** IN/OUT ARGS : xInit, xEnd, partitions                                       ***
*** RETURN      : void                                                          ***
**********************************************************************************/
void ReadCommandLine(int argc, char* argv[], double* xInit, double* xEnd, unsigned long* numParts)
{
	*xInit = atof(argv[1]);
	*xEnd = atof(argv[2]);
	sscanf(argv[3], "%lu", numParts);
}

/**********************************************************************************
*** FUNCTION midpoint                                                           ***
***********************************************************************************
*** DESCRIPTION : Calculates the integral from start->end of function calcFx    ***
				  Uses riemann rectangles evaluated at the midpoint.
*** INPUT ARGS  : start, end, partitions, width                                 ***
*** OUTPUT ARGS : none                                                          ***
*** IN/OUT ARGS : none                                                          ***
*** RETURN      : double                                                        ***
**********************************************************************************/
double midpoint(double start, unsigned long partitions, double width, int* num_threads)
{
	unsigned long i;
	double x, half_width = width / 2.0, total = 0;
	
	#pragma omp parallel default(none) private(i,x) shared(num_threads, start, partitions, width, half_width) reduction(+:total)
	{
        *num_threads = omp_get_num_threads();
		
		#pragma omp for
		for(i = 0; i < partitions; i++)
		{
			x = start + half_width + i * width;
			total += calcFx(x);
		}
		total *= width * 4.0;
	}
	return total;
}

/**********************************************************************************
*** FUNCTION calcFx                                                             ***
***********************************************************************************
*** DESCRIPTION : Returns the value of f(x) at x
                  f(x) = 1/(1+x^2)                                              ***
*** INPUT ARGS  : x                                                             ***
*** OUTPUT ARGS : none                                                          ***
*** IN/OUT ARGS : none                                                          ***
*** RETURN      : double                                                        ***
**********************************************************************************/
double calcFx(double x)
{
    return 1.0/(1.0+ x*x);
}

/**********************************************************************************
*** FUNCTION display                                                            ***
***********************************************************************************
*** DESCRIPTION : Displays output                                               ***
*** INPUT ARGS  : total, xinit, xEnd, partitions, time, comm_size, num_threads  ***
*** OUTPUT ARGS : none                                                          ***
*** IN/OUT ARGS : none                                                          ***
*** RETURN      : void                                                          ***
**********************************************************************************/
void display(double total, double xInit, double xEnd, unsigned long partitions, double time, int comm_size, int num_threads)
{
	double pi;
    double diff = total - PI;
	pi = PI;

	printf("Starting x          : %*f\n", 26, xInit);
	printf("Ending x            : %*f\n", 26, xEnd);
	printf("Total Processes     : %*d\n", 26, comm_size);
	printf("Partitions/Process  : %*lu\n", 26, partitions);
	printf("Total Partitions    : %26.1lu\n", partitions*comm_size);
	printf("Threads/Process     : %*d\n\n", 26, num_threads);

	printf("Real Pi             : %26.15f\n", pi);
    printf("Calculated Pi       : %26.15f\n", total);
    printf("Difference          : %26.15f\n", diff);
	printf("Time to calculate   : %26.15f\n\n", time);
}
