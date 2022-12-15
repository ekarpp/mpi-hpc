#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

int main(int argc, char **argv)
{
    char *ptr;
    int DIM = 1 << strtol(argv[1], &ptr, 10);

    MPI_Init(&argc, &argv);
    int nprocs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int dim_fix = ((DIM % nprocs) == 0)
        ? DIM
        : DIM + (nprocs - (DIM % nprocs));
    int size = dim_fix * dim_fix;
    int chunk = dim_fix / nprocs;

    double *data = NULL;
    double *datat = NULL;

    if (rank == 0)
    {
        data = malloc(sizeof(double) * size);
        datat = malloc(sizeof(double) * size);
        for (int i = 0; i < dim_fix; i++)
            for (int j = 0; j < dim_fix; j++)
                data[i*dim_fix + j] = (j < DIM && i < DIM)
                    ? drand48() + 0.01
                    : 0.0f;
    }


    /* performs the transposition with custom data types */
    /* send_t = read columns */
    /* recv_t = write rows */
    MPI_Datatype tmp, send_t, recv_t;
    MPI_Type_vector(dim_fix, chunk, dim_fix, MPI_DOUBLE, &tmp);
    MPI_Type_create_resized(tmp, 0, chunk*sizeof(double), &send_t);
    MPI_Type_commit(&send_t);

    MPI_Type_vector(chunk, 1, dim_fix, MPI_DOUBLE, &tmp);
    MPI_Type_create_resized(tmp, 0, sizeof(double), &recv_t);
    MPI_Type_commit(&recv_t);

    double *subm = calloc(dim_fix * chunk, sizeof(double));
    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();

    /* just scatter and gather now :) */
    MPI_Scatter(
        data, 1,       send_t,
        subm, dim_fix, recv_t,
        0, MPI_COMM_WORLD
    );
    MPI_Gather(
        subm,  dim_fix * chunk, MPI_DOUBLE,
        datat, dim_fix * chunk, MPI_DOUBLE,
        0, MPI_COMM_WORLD
    );

    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime() - t;

    free(subm);

    if (rank == 0)
    {
        printf("%d x %d matrix in %.4f s with %d proc\n", DIM, DIM, t, nprocs);
        t = MPI_Wtime();
        for (int i = 0; i < dim_fix; i++)
        {
            for (int j = i; j < dim_fix; j++)
            {
                if (data[i*dim_fix + j] != datat[j*dim_fix + i])
                    printf("err at (%d,%d)\n", i, j);
            }
        }
        t = MPI_Wtime() - t;
        printf("verification in %.4f s\n", t);
        free(data);
        free(datat);
    }

    MPI_Type_free(&recv_t);
    MPI_Type_free(&send_t);
    MPI_Finalize();
    return 0;
}
