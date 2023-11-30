#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

//#define MATRIX_SIZE 4

const char* luDecompositionKernelSource =
"__kernel void luDecomposition(__global float* A, __global float* L, __global float* U, int width) {\n"
"    int gidX = get_global_id(0);\n"
"    int gidY = get_global_id(1);\n"
"    if (gidX < width && gidY < width) {\n"
"        if (gidY <= gidX) {\n"
"            float sum = 0.0f;\n"
"            for (int k = 0; k < gidY; ++k) {\n"
"                sum += L[gidX * width + k] * U[k * width + gidY];\n"
"            }\n"
"            U[gidY * width + gidX] = (gidX == gidY) ? 1.0f : (A[gidY * width + gidX] - sum);\n"
"        }\n"
"        if (gidY >= gidX) {\n"
"            float sum = 0.0f;\n"
"            for (int k = 0; k < gidX; ++k) {\n"
"                sum += L[gidY * width + k] * U[k * width + gidX];\n"
"            }\n"
"            L[gidY * width + gidX] = (A[gidY * width + gidX] );\n"
"        }\n"
"    }\n"
"}\n";



void printMatrix(const char* name, float* matrix, int rows, int cols) {
    printf("%s:\n", name);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            printf("%f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

void matrixMultiply(float* L, float* U, float* result, int rows, int cols) {
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
        {
            float sum = 0;
            for (int k = 0; k < cols; k++)
                sum += L[i * cols + k] * U[k * cols + j];
            result[i * cols + j] = sum;
        }   
}


void solveLinearSystem(float* L, float* U, float* B, float* result, int size) {
    float** y_thread = (float**)malloc(sizeof(float*) * omp_get_max_threads());
#pragma omp parallel
    {
        int thread_id = omp_get_thread_num();
        y_thread[thread_id] = (float*)malloc(sizeof(float) * size);

#pragma omp for
        for (int i = 0; i < size; ++i) {
            float sum = 0.0f;
            for (int j = 0; j < i; ++j) {
                sum += L[i * size + j] * y_thread[thread_id][j];
            }
            y_thread[thread_id][i] = (B[i] - sum) / L[i * size + i];
        }

#pragma omp barrier

#pragma omp for
        for (int i = size - 1; i >= 0; --i) {
            float sum = 0.0f;
            for (int j = i + 1; j < size; ++j) {
                sum += U[i * size + j] * result[j];
            }
            result[i] = (y_thread[thread_id][i] - sum) / U[i * size + i];
        }

        free(y_thread[thread_id]);
    }

    free(y_thread);
}

float computeDeterminant(float* U, int size) {
    float determinant = 1.0f;
    for (int i = 0; i < size; ++i) {
        determinant *= U[i * size + i];
    }
    return determinant;
}



int main() {
    cl_platform_id platform;
    clGetPlatformIDs(1, &platform, NULL);

    cl_device_id device;
    clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);

    cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
    cl_command_queue queue = clCreateCommandQueueWithProperties(context, device, NULL, NULL);
    // Ввод размера матрицы
    int MATRIX_SIZE;
    printf("Enter the size of the matrix A: ");
    scanf_s("%d", &MATRIX_SIZE);

    // Динамическое выделение памяти для матрицы A и вектора B
    float* matrixA = (float*)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    float* matrixB = (float*)malloc(sizeof(float) * MATRIX_SIZE);

    // Ввод матрицы A
    printf("Enter the matrix A (%dx%d):\n", MATRIX_SIZE, MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            printf("A[%d][%d]: ", i, j);
            scanf_s("%f", &matrixA[i * MATRIX_SIZE + j]);
        }
    }

    // Ввод вектора B
    printf("Enter the vector B (%d elements):\n", MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        printf("B[%d]: ", i);
        scanf_s("%f", &matrixB[i]);
    }

    printMatrix("Matrix A", matrixA, MATRIX_SIZE, MATRIX_SIZE);
    int matrixSize = MATRIX_SIZE;
    cl_mem bufferA = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, matrixA, NULL);


    cl_mem bufferMatrixSize = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
        sizeof(int), &matrixSize, NULL);


    cl_program luDecompositionProgram = clCreateProgramWithSource(context, 1, &luDecompositionKernelSource, NULL, NULL);
    clBuildProgram(luDecompositionProgram, 1, &device, NULL, NULL, NULL);

    cl_kernel kernelLU = clCreateKernel(luDecompositionProgram, "luDecomposition", NULL);

    // Выделение памяти для L и U матриц
    float* matrixL = (float*)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    float* matrixU = (float*)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    //float matrixL[MATRIX_SIZE * MATRIX_SIZE];
    //float matrixU[MATRIX_SIZE * MATRIX_SIZE];

    cl_mem bufferL = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);
    cl_mem bufferU = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);

    // Выполнение LU-разложения матрицы на GPU
    clSetKernelArg(kernelLU, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernelLU, 1, sizeof(cl_mem), &bufferL);
    clSetKernelArg(kernelLU, 2, sizeof(cl_mem), &bufferU);
    clSetKernelArg(kernelLU, 3, sizeof(int), &matrixSize);

    size_t globalSizeLU[2] = { MATRIX_SIZE, MATRIX_SIZE };
    clEnqueueNDRangeKernel(queue, kernelLU, 2, NULL, globalSizeLU, NULL, 0, NULL, NULL);

    clEnqueueReadBuffer(queue, bufferL, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, matrixL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, bufferU, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, matrixU, 0, NULL, NULL);


    printMatrix("Matrix L", matrixL, MATRIX_SIZE, MATRIX_SIZE);
    printMatrix("Matrix U", matrixU, MATRIX_SIZE, MATRIX_SIZE);

    float* matrixResult = (float*)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    //float matrixResult[MATRIX_SIZE * MATRIX_SIZE];
    matrixMultiply(matrixL, matrixU, matrixResult, MATRIX_SIZE, MATRIX_SIZE);

    printMatrix("Matrix Result (L * U)", matrixResult, MATRIX_SIZE, MATRIX_SIZE);

    // Решение системы линейных уравнений Ax = B
    float* solution = (float*)malloc(sizeof(float) * MATRIX_SIZE * MATRIX_SIZE);
    //float solution[MATRIX_SIZE];

    solveLinearSystem(matrixL, matrixU, matrixB, solution, MATRIX_SIZE);

    // Проверка невырожденности матрицы A
    float determinant = computeDeterminant(matrixU, MATRIX_SIZE);
    if (determinant != 0.0f) {
        printf("Matrix A is non-singular (det(A) != 0)\n");

        printf("Solution of Ax = B:\n");
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            printf("%f\n", solution[i]);
        }
    }
    else {
        printf("Matrix A is singular (det(A) = 0), cannot solve the system.\n");
    }

    // Освобождение памяти
    clReleaseMemObject(bufferL);
    clReleaseMemObject(bufferU);
    clReleaseKernel(kernelLU);
    clReleaseProgram(luDecompositionProgram);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    return 0;
}
