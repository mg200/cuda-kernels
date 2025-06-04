// Function to read the mask from a file
float** read_mask(const char* mask_file, int* mask_dimension) {
    // Open the file for reading
    FILE* file = fopen(mask_file, "r");
    if (file == NULL) {
        fprintf(stderr, "Error: Unable to open mask file %s\n", mask_file);
        exit(1);
    }

    // Read the mask dimension (N)
    if (fscanf(file, "%d", mask_dimension) != 1) {
        fprintf(stderr, "Error: Unable to read mask dimension from file %s\n", mask_file);
        fclose(file);
        exit(1);
    }

    // Allocate memory for the mask
    float** mask = (float**)malloc((*mask_dimension) * sizeof(float*));
    if (mask == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for mask\n");
        fclose(file);
        exit(1);
    }

    for (int i = 0; i < *mask_dimension; ++i) {
        mask[i] = (float*)malloc((*mask_dimension) * sizeof(float));
        if (mask[i] == NULL) {
            fprintf(stderr, "Error: Memory allocation failed for mask row %d\n", i);
            fclose(file);
            for (int j = 0; j < i; ++j) {
                free(mask[j]);
            }
            free(mask);
            exit(1);
        }
    }

    // Read the mask values
    for (int i = 0; i < *mask_dimension; ++i) {
        for (int j = 0; j < *mask_dimension; ++j) {
            if (fscanf(file, "%f", &mask[i][j]) != 1) {
                fprintf(stderr, "Error: Unable to read mask value at [%d][%d] from file %s\n", i, j, mask_file);
                fclose(file);
                for (int k = 0; k < *mask_dimension; ++k) {
                    free(mask[k]);
                }
                free(mask);
                exit(1);
            }
        }
    }

    
    fclose(file);
    return mask;
}

// Flatten the 2D mask into a 1D array
float* flatten_mask(float** mask, int dimension) {
    float* flattened = (float*)malloc(dimension * dimension * sizeof(float));
    if (flattened == NULL) {
        fprintf(stderr, "Error: Memory allocation failed for flattened mask\n");
        exit(1);
    }
    for (int i = 0; i < dimension; ++i) {
        for (int j = 0; j < dimension; ++j) {
            flattened[i * dimension + j] = mask[i][j];
        }
    }
    return flattened;
}