#include <mpi.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

#define MIN(X,Y) ((X > Y) ? Y : X)
#define MAX(X,Y) ((X > Y) ? X : Y)

/* return the index of mediam of three. evil xor hacks */
int median(uint64_t *data, int i, int j, int k)
{
    const uint64_t ii = data[i];
    const uint64_t jj = data[j];
    const uint64_t kk = data[k];

    if ((ii > jj) ^ (ii > kk))
        return i;
    else if ((jj < ii) ^ (jj < kk))
        return j;
    else
        return k;
}

/* usual partition with median of three pivot */
int partition(uint64_t *data, int len)
{
    const int pivot_idx = median(data, 0, len / 2, len - 1);
    const uint64_t pivot = data[pivot_idx];
    data[pivot_idx] = data[len - 1];
    data[len - 1] = pivot;
    int pivot_pos = 0;

    for (int i = 0; i < len; i++)
    {
        if (data[i] < pivot)
        {
            const uint64_t tmp = data[i];
            data[i] = data[pivot_pos];
            data[pivot_pos] = tmp;
            pivot_pos++;
        }
    }
    data[len - 1] = data[pivot_pos];
    data[pivot_pos] = pivot;

    return pivot_pos;
}

/* for c stdlib qsort */
int cmpfunc (const void *a, const void *b) {
    return ( *(uint64_t*)a > *(uint64_t*)b ) ? 1 : 0;
}

void handle_zero(uint64_t *data, int n, int *nbors, MPI_Comm comm)
{
    MPI_Request req[6];
    const int len = 0;
    for (int i = 0; i < n; i++)
    {
        MPI_Isend(&len, 1, MPI_INT, nbors[i], 0, comm, &req[3*i + 0]);
        MPI_Isend(data, 0, MPI_UINT64_T, nbors[i], 1, comm, &req[3*i + 1]);
        MPI_Irecv(data, 0, MPI_UINT64_T, nbors[i], 0, comm, &req[3*i + 2]);
    }
    MPI_Waitall(3*n, req, MPI_STATUSES_IGNORE);
}

void sort_mpi(uint64_t *data, int len, int n, int *nbors, MPI_Comm comm)
{
    if (len == 0)
        return handle_zero(data, n, nbors, comm);
    if (n == 0)
        return qsort(data, len, sizeof(uint64_t), cmpfunc);

    const int len_data = partition(data, len);
    const int len_rest = len - len_data - 1;
    uint64_t *rest = data + len_data + 1;
    if (n == 1)
    {
        /* send the shorter partition to the remaining process */
        const int send_n = MIN(len_rest, len_data);
        uint64_t *send_buf = (len_rest > len_data)
            ? data
            : rest;

        MPI_Request req[3];
        MPI_Isend(&send_n, 1, MPI_INT, nbors[0], 0, comm, &req[0]);
        MPI_Isend(send_buf, send_n, MPI_UINT64_T, nbors[0], 1, comm, &req[1]);

        MPI_Irecv(send_buf, send_n, MPI_UINT64_T, nbors[0], 0, comm, &req[2]);

        /* sort the longer partition with qsort */
        if (len_rest > len_data)
            qsort(rest, len_rest, sizeof(uint64_t), cmpfunc);
        else
            qsort(data, len_data, sizeof(uint64_t), cmpfunc);

        MPI_Waitall(3, req, MPI_STATUSES_IGNORE);
    }
    else
    {
        MPI_Request req[6];
        MPI_Isend(&len_data, 1, MPI_INT, nbors[0], 0, comm, &req[0]);
        MPI_Isend(data, len_data, MPI_UINT64_T, nbors[0], 1, comm, &req[1]);

        MPI_Irecv(data, len_data, MPI_UINT64_T, nbors[0], 0, comm, &req[2]);

        MPI_Isend(&len_rest, 1, MPI_INT, nbors[1], 0, comm, &req[3]);
        MPI_Isend(rest, len_rest, MPI_UINT64_T, nbors[1], 1, comm, &req[4]);

        MPI_Irecv(rest, len_rest, MPI_UINT64_T, nbors[1], 0, comm, &req[5]);

        MPI_Waitall(6, req, MPI_STATUSES_IGNORE);
    }
}

uint64_t *mpi_qs(MPI_Comm comm, uint64_t *data, int rank, int len)
{
    int n;
    MPI_Graph_neighbors_count(comm, rank, &n);
    int *nbors = malloc(sizeof(int) * n);
    MPI_Graph_neighbors(comm, rank, n, nbors);

    /* rank of parent in the b tree.
     * if root, no need to receive send/data here */
    const int parent = (rank == 0)
        ? MPI_PROC_NULL
        : (rank - ((rank+1) % 2)) >> 1;

    MPI_Recv(&len, 1, MPI_INT, parent, 0, comm, MPI_STATUS_IGNORE);
    /* only root has allocated data */
    if (data == NULL)
        data = malloc(sizeof(uint64_t) * len);
    MPI_Recv(data, len, MPI_UINT64_T, parent, 1, comm, MPI_STATUS_IGNORE);

    sort_mpi(data, len, n, nbors, comm);

    MPI_Send(data, len, MPI_UINT64_T, parent, 0, comm);
    free(nbors);
    return data;
}

/* use the naive quicksort, partition data and send partitions to childs.
 * if not childs remaining, call c stdlibs qsort */
int main(int argc, char **argv)
{
    char *ptr;
    const int len = 1 << strtol(argv[1], &ptr, 10);

    MPI_Init(&argc, &argv);
    int nprocs;
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    /* create B-tree graph topology */
    int *index = calloc(nprocs, sizeof(int));
    int *edges = malloc(sizeof(int) * (nprocs - 1));
    for (int i = 1; i < nprocs; i++)
        edges[i - 1] = i;

    /* compute degrees of nodes for the b tree */
    int child_left = nprocs - 1;
    int idx = 0;
    while (child_left > 0)
    {
        const int childs = MIN(child_left, 2);
        index[idx] = childs;
        child_left -= childs;
        idx++;
    }
    /* mpi wants a "rolling sum".
     * index calloced to all zeros, we good here. */
    for (int i = 1; i < nprocs; i++)
        index[i] += index[i - 1];

    MPI_Comm b_tree;
    int rank;
    MPI_Graph_create(MPI_COMM_WORLD, nprocs, index, edges, 0, &b_tree);
    MPI_Comm_rank(b_tree, &rank);

    uint64_t *data = NULL;
    if (rank == 0)
    {
        data = malloc(sizeof(uint64_t) * len);
        for (int i = 0; i < len; i++)
            data[i] = rand();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    double t = MPI_Wtime();
    data = mpi_qs(b_tree, data, rank, len);
    MPI_Barrier(MPI_COMM_WORLD);
    t = MPI_Wtime() - t;

    if (rank == 0)
    {
        printf("%d elements in %f s with %d tasks\n", len, t, nprocs);
        for (int i = 0; i < len - 1; i++)
            if (data[i] > data[i+1])
                printf("err at %d\n", i);
    }

    free(data);
    free(index);
    free(edges);
    MPI_Comm_free(&b_tree);
    MPI_Finalize();
    return 0;
}
