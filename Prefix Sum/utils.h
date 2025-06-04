#include <stdio.h>
#include <stdlib.h>

#include <cuda.h>
#include <assert.h>
#include <math.h>
#include <cmath>
#include <iostream>
#include <string>

float **read_file(const char *filename, int *rows, int *cols)
{
    // Open the file for reading
    FILE *file = fopen(filename, "r");
    if (file == NULL)
    {
        fprintf(stderr, "Error: Unable to open file %s\n", filename);
        exit(1);
    }

    // Read matrix dimensions
    if (fscanf(file, "%d %d", rows, cols) != 2)
    {
        fprintf(stderr, "Error: Unable to read matrix dimensions from file %s\n", filename);
        fclose(file);
        exit(1);
    }

    // Allocate memory for the matrix
    float **matrix = (float **)malloc(*rows * sizeof(float *));
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed for matrix\n");
        fclose(file);
        exit(1);
    }

    for (int i = 0; i < *rows; ++i)
    {
        matrix[i] = (float *)malloc(*cols * sizeof(float));
        if (matrix[i] == NULL)
        {
            fprintf(stderr, "Error: Memory allocation failed for row %d\n", i);
            for (int j = 0; j < i; ++j)
            {
                free(matrix[j]);
            }
            free(matrix);
            fclose(file);
            exit(1);
        }
    }

    // Read matrix values
    for (int i = 0; i < *rows; ++i)
    {
        for (int j = 0; j < *cols; ++j)
        {
            if (fscanf(file, "%f", &matrix[i][j]) != 1)
            {
                fprintf(stderr, "Error: Unable to read value at [%d][%d] from file %s\n", i, j, filename);
                for (int k = 0; k <= i; ++k)
                {
                    free(matrix[k]);
                }
                free(matrix);
                fclose(file);
                exit(1);
            }
        }
    }

    fclose(file);
    return matrix;
}

void print_matrix(float **matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            printf("%6.2f ", matrix[i][j]);
        }
        printf("\n");
    }
}

void print_matrix(float *matrix, int rows, int cols)
{
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < cols; ++j)
        {
            printf("%6.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void print_arr(float *matrix, int size)
{
    for (int j = 0; j < size; ++j)
    {
        printf("%6.2f ", matrix[j]);
    }
    printf("\n");
}

template<typename T>
T *readArr(FILE *file, int *size)
{
    if (fscanf(file, "%d", size) != 1)
    {
        fprintf(stderr, "Error: Failed to read file dimensions.\n");
        return NULL;
    }
    int temp = (*size);
    T *mat = (T *)malloc(sizeof(T) * temp);
    if (mat == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return NULL;
    }
    for (int i = 0; i < temp; ++i)
    {
        if constexpr (std::is_same<T, double>::value)
        {
            if (fscanf(file, "%lf", &mat[i]) != 1)
            {
                fprintf(stderr, "Error: Failed to read double value at index %d.\n", i);
                free(mat);
                return NULL;
            }
        }
        else
        {
            if (fscanf(file, "%f", &mat[i]) != 1)
            {
                fprintf(stderr, "Error: Failed to read float value at index %d.\n", i);
                free(mat);
                return NULL;
            }
        }
    }
    return mat;
}


void readArr(FILE* f, double* arr, int size) {
    for (int i = 0; i < size; ++i) {
        if (fscanf(f, "%lf", &arr[i]) != 1) {
            fprintf(stderr, "Error reading element %d from file.\n", i);
            exit(1);
        }
    }
}

void readMat(FILE *file, float *mat, int rows, int cols)
{
    for (int i = 0; i < rows * cols; ++i)
    {
        fscanf(file, "%f", &mat[i]);
    }
}

float *readMat(FILE *file, int *rows, int *cols)
{
    if (fscanf(file, "%d %d", rows, cols) != 2)
    {
        fprintf(stderr, "Error: Failed to read matrix dimensions.\n");
        return NULL;
    }

    int total = (*rows) * (*cols);
    float *mat = (float *)malloc(total * sizeof(float));
    if (mat == NULL)
    {
        fprintf(stderr, "Error: Memory allocation failed.\n");
        return NULL;
    }

    for (int i = 0; i < total; ++i)
    {
        if (fscanf(file, "%f", &mat[i]) != 1)
        {
            fprintf(stderr, "Error: Failed to read matrix data at index %d.\n", i);
            free(mat);
            return NULL;
        }
    }

    return mat;
}
float **allocate_matrix(int rows, int cols)
{
    float **matrix = (float **)malloc(rows * sizeof(float *));
    if (matrix == NULL)
    {
        fprintf(stderr, "Error: Could not allocate memory for row pointers\n");
        return NULL;
    }

    for (int i = 0; i < rows; ++i)
    {
        matrix[i] = (float *)malloc(cols * sizeof(float));
        if (matrix[i] == NULL)
        {
            fprintf(stderr, "Error: Could not allocate memory for row %d\n", i);
            // Free already allocated rows to avoid memory leaks
            for (int j = 0; j < i; ++j)
            {
                free(matrix[j]);
            }
            free(matrix);
            return NULL;
        }
    }

    return matrix;
}
