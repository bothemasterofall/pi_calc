

/**********************************************************************
*** NAME           : Bo Cimino                                      ***
*** CLASS          : CSc 318                                        ***
*** DUE DATE       : 04/02/2014                                     ***
*** INSTRUCTUOR    : GAMRADT                                        ***
***********************************************************************
*** DESCRIPTION: This is the 5th assignment for Parallel
                 Programming. This uses a riemann sum to calculate pi,
                 should the user's interval be from 0 to 1; The function
                 is hardcoded in. The objective here is to compare different
                 performance obtained by adding more threads to do the work.
                 Data parallelism is what is implemented here.
                 This assignment difers from 4 because of more detailed
                 omp clauses. In addition we use a parallel for directive.
**********************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

const double pi = 3.1415926535897932384626433;

void welcome();
void ReadCommandLine(const int argc, const char* argv[], double* xInit, double* xEnd, unsigned int* numThreads, unsigned long* numParts);
void calcPi(double xInit, double xEnd, unsigned long partitions, unsigned int threads, double* glob_result);
double calcFx(double x);
void display(double myPie, double time, unsigned long partitions);


int main(const int argc, const char* argv[])
{
    unsigned int threads;
    unsigned long partitions;
    double myPie = 0.0, current_time, ti, tf, xInit, xEnd;

    welcome();
    ReadCommandLine(argc, argv, &xInit, &xEnd, &threads, &partitions);

    system("clear");

    ti = omp_get_wtime();
    calcPi(xInit, xEnd, partitions, threads, &myPie);
    tf = omp_get_wtime();
    current_time = tf-ti;

    display(myPie, current_time, partitions);

    return 0;
}

void welcome()
{
    printf("This program will, if your range is from 0 to 1, calculate pi.\n");
    printf("If all your arguments were not entered via command line,\nyou will be prompted.\n\n");
}

/**********************************************************************************
*** FUNCTION ReadCommandLine                                                    ***
***********************************************************************************
*** DESCRIPTION : This function reads in input from the command line. In this program
                  this is the number of partitions desired.                     ***
*** INPUT ARGS  : argc, argv                                                    ***
*** OUTPUT ARGS : none                                                          ***
*** IN/OUT ARGS : partitions                                                    ***
*** RETURN      : void                                                          ***
**********************************************************************************/
void ReadCommandLine(const int argc, const char* argv[], double* xInit, double* xEnd, unsigned int* numThreads, unsigned long* numParts)
{
    if(argc != 5)
    {
        printf("Enter starting point -> ");
        scanf("%lf", xInit);
        printf("Enter ending point -> ");
        scanf("%lf", xEnd);
        printf("Enter no. of threads -> ");
        scanf("%ud", numThreads);
        printf("Enter no. of partitions -> ");
        scanf("%lu", numParts);
    }

    else
    {
        *xInit = atof(argv[1]);
        *xEnd = atof(argv[2]);
        *numThreads = atoi(argv[3]);
        *numParts = atol(argv[4]);
    }
}

/**********************************************************************************
*** FUNCTION calcPi                                                             ***
***********************************************************************************
*** DESCRIPTION : Calculates pi using the area under the curve based on the function
                  in calcFx on the interval [0, 1].                             ***
*** INPUT ARGS  : partitions                                                    ***
*** OUTPUT ARGS : none                                                          ***
*** IN/OUT ARGS : none                                                          ***
*** RETURN      : double                                                        ***
**********************************************************************************/
void calcPi(double xInit, double xEnd, unsigned long partitions, unsigned int threads, double* glob_total)
{
    unsigned long i;
    double local_total, width, half_width, x;

    width = (xEnd-xInit)/ (float) partitions;
    half_width = width / 2.0;

    local_total = 0;

    #pragma omp parallel num_threads(threads) default(none) private(i,x) shared(partitions, width, half_width) reduction(+:local_total)
    {
        int thread_num = omp_get_thread_num();
        int num_threads = omp_get_num_threads();

        #pragma omp for
        for(i = 0; i < partitions; i++)
        {
            x = half_width + i * width;
            local_total += calcFx(x);
        }
        local_total = local_total * width * 4.0;

        #pragma omp critical
        {
            printf("Thread %d of %d total: % 25.15f\n", thread_num, num_threads, local_total);
        }
    }
    *glob_total = (local_total);
}

/**********************************************************************************
*** FUNCTION calcFx                                                             ***
***********************************************************************************
*** DESCRIPTION : Returns the value of f(x) at x
                  f(x) = 1/(1+x^2)                                              ***
*** INPUT ARGS  : x                                                             ***
*** OUTPUT ARGS : none                                                          ***
*** IN/OUT ARGS : none                                                          ***
*** RETURN      : float                                                         ***
**********************************************************************************/
double calcFx(double x)
{
    return 1/(1+ x*x);
}

/**********************************************************************************
*** FUNCTION display                                                            ***
***********************************************************************************
*** DESCRIPTION : Displays output                                               ***
*** INPUT ARGS  : myPie, time, partitions                                       ***
*** OUTPUT ARGS : none                                                          ***
*** IN/OUT ARGS : none                                                          ***
*** RETURN      : void                                                          ***
**********************************************************************************/
void display(double myPie, double time, unsigned long partitions)
{
    double diff = myPie-pi;

    printf("Partitions  : %16.1lu\n\n", partitions);

    printf("Real Pi             : %26.15f\n", pi);
    printf("Calculated Pi       : %26.15f\n", myPie);
    printf("Difference          : %26.15f\n\n", fabs(diff));
    printf("Time to calculate   : %26.15f\n\n", time);
}

