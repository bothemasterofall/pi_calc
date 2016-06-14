/*******************************************************************************
*   NAME           : Bo Cimino
*   CLASS          : CSc 318
*   DUE DATE       : 04/21/2014
*   INSTRUCTUOR    : GAMRADT
********************************************************************************
*   DESCRIPTION:
*       This program calculates pi by integrating the function 1/(1+x^2) on the
*       interval 0->1 using a riemann sum.  Because millions or billions of
*       partitions are used in this riemann sum, parallel programming techniques
*       are implemented using mpi.  At the end, the approximate value of pi is
*       output, along with the real value of pi and the time it took to
*       calculate it.
*******************************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

const double pi = 3.1415926535897932384626433;

void welcome();
void read_command_line(int argc, char* argv[], double* x_init, double* x_end,
    unsigned long* numParts);
void calc_pi(double x_init, double x_end, unsigned long partitions,
    double* glob_total, int my_rank, int comm_size);
double midpoint(double start, unsigned long partitions, double width);
double calc_fx(double x);
void display(double my_pie, unsigned long partitions, double time);

int main(int argc, char* argv[])
{
    unsigned long partitions = 0;
    double my_pie = 0.0, x_init = 0, x_end = 1, ti, tf, tr;
    int my_rank, comm_size;

    /*Initialize mpi*/
    MPI_Init(&argc, &argv);
    MPI_Comm_rank( MPI_COMM_WORLD, &my_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);

    if(my_rank == 0)
    {
        welcome();
    }

    read_command_line(argc, argv, &x_init, &x_end, &partitions);

    if(my_rank == 0)
    {
        ti = MPI_Wtime();
    }

    calc_pi(x_init, x_end, partitions, &my_pie, my_rank, comm_size);    

    if(my_rank == 0)
    {
        tf = MPI_Wtime();
        tr = tf - ti;
        display(my_pie, partitions, tr);
    }
    MPI_Finalize();
    return 0;
}

/*******************************************************************************
*   FUNCTION calc_fx
********************************************************************************
*** DESCRIPTION : Welcomes the user and explains the program to them
*** INPUT ARGS  : none
*** OUTPUT ARGS : none
*** IN/OUT ARGS : none
*** RETURN      : void
*******************************************************************************/
void welcome()
{
    printf("This program will calculate pi.\n");
    printf("The function 1/(1+x^2) is evaluated from 0 to 1 using a Riemann "
        "sum.\n");
    printf("The number of rectangles that the function is divided into is "
        "specified\nin the pbs file.\n");
    printf("Because millions (or billions) of partitions are used,\n");
    printf("parallelism is implemented using mpi to speed up the "
        "computation.\n\n");
}

/*******************************************************************************
*   FUNCTION read_command_line
********************************************************************************
*   DESCRIPTION : This function reads in input from the command line. In this
*       program this is the number of partitions desired.
*   INPUT ARGS  : argc, argv
*   OUTPUT ARGS : none
*   IN/OUT ARGS : x_init, x_end, partitions
*   RETURN      : void
*******************************************************************************/
void read_command_line(int argc, char* argv[], double* x_init, double* x_end,
    unsigned long* numParts)
{
    *x_init = atof(argv[1]);
    *x_end = atof(argv[2]);
    *numParts = atol(argv[3]);
}

/*******************************************************************************
*   FUNCTION calc_pi
********************************************************************************
*   DESCRIPTION : Calculates pi using the area under the curve based on the
*       function in calc_fx on the interval [0, 1].
*   INPUT ARGS  : x_init, x_end, partitions, my_rank, comm_size
*   OUTPUT ARGS : none
*   IN/OUT ARGS : glob_total
*   RETURN      : void
*******************************************************************************/
void calc_pi(double x_init, double x_end, unsigned long partitions,
    double* glob_total, int my_rank, int comm_size)
{    
    int local_partitions;
    double local_start, width, half_width, process_width;
    double local_integral = 0, total_integral = 0;
    int source;

    /*common to all processes*/
    width = (x_end-x_init)/partitions;
    half_width = width/2;
    local_partitions = partitions / comm_size; /*parts per process */
    process_width = local_partitions * width;

    /*unique to each process*/
    local_start = x_init + my_rank * process_width;
    local_integral = midpoint(local_start, local_partitions, width);

    if(my_rank != 0)
    {
        /*send local_integral to process 0*/
        printf("Process %d total = %1.5f\n", my_rank, local_integral);
        MPI_Send(&local_integral, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }
    else /*my_rank == 0*/
    {
        total_integral = local_integral;
        for(source = 1; source < comm_size; source++)
        {
            /* receive local_integral from process i - i == 1 to n-1*/
            MPI_Recv(&local_integral, 1, MPI_DOUBLE, source, 0, MPI_COMM_WORLD,
                MPI_STATUS_IGNORE);
            total_integral += local_integral;
        }
    }
    if(my_rank == 0)
    {
        *glob_total = total_integral;
    }
}

/*******************************************************************************
*   FUNCTION midpoint
********************************************************************************
*   DESCRIPTION : Calculates the integral from start->end of function calc_fx.
*       Uses riemann rectangles evaluated at the midpoint.
*   INPUT ARGS  : start, end, partitions, width
*   OUTPUT ARGS : none
*   IN/OUT ARGS : none
*   RETURN      : double
*******************************************************************************/
double midpoint(double start, unsigned long partitions, double width)
{
    int i = 0;
    double x, half_width = width / 2.0, total = 0;

    for(i = 0; i < partitions; i++)
    {
        x = start + half_width + i * width;
        total += calc_fx(x);
    }
    total *= width;
    total *= 4;
    return total;
}

/*******************************************************************************
*   FUNCTION calc_fx
********************************************************************************
*   DESCRIPTION : Returns the value of f(x) at x where
*       f(x) = 1/(1+x^2)
*   INPUT ARGS  : x
*   OUTPUT ARGS : none
*   IN/OUT ARGS : none
*   RETURN      : double
*******************************************************************************/
double calc_fx(double x)
{
    return 1/(1+ x*x);
}

/*******************************************************************************
*   FUNCTION display
********************************************************************************
*   DESCRIPTION : Displays output
*   INPUT ARGS  : my_pie, time, partitions
*   OUTPUT ARGS : none
*   IN/OUT ARGS : none
*   RETURN      : void
*******************************************************************************/
void display(double my_pie, unsigned long partitions, double time)
{
    double diff = my_pie-pi;

    printf("Partitions          : %26.1lu\n\n", partitions);

    printf("Real Pi             : %26.15f\n", pi);
    printf("Calculated Pi       : %26.15f\n", my_pie);
    printf("Difference          : %26.15f\n", diff);
    printf("Time to calculate   : %26.15f\n\n", time);
}
