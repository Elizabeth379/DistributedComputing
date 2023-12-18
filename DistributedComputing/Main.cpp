#include <CL/cl.h>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <cmath>


const char* luDecompositionKernelSource =
"__kernel void luDecomposition(__global float* A, __global float* L, __global float* U, const int N) {\n"
"    int i, j, k;\n"
"\n"
"    for (k = 0; k < N; k++) {\n"
"        float Akk = A[k * N + k];\n"
"        L[k * N + k] = 1.0f;\n"
"        U[k * N + k] = Akk;\n"
"\n"
"        for (i = k + 1; i < N; i++) {\n"
"            L[i * N + k] = A[i * N + k] / Akk;\n"
"            U[k * N + i] = A[k * N + i];\n"
"        }\n"
"\n"
"        for (i = k + 1; i < N; i++) {\n"
"            for (j = k + 1; j < N; j++) {\n"
"                A[i * N + j] -= L[i * N + k] * U[k * N + j];\n"
"            }\n"
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
#pragma omp parallel for collapse(3)
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

// Функция для вычисления определителя на основе матрицы U с распараллеливанием
long double determinantFromLU(float* matrixU, int size) {
    long double det = 1.0f;

#pragma omp parallel for reduction(*:det) num_threads(omp_get_max_threads())
    for (int i = 0; i < size; ++i) {
        det *= matrixU[i * size + i];
    }

    return det;
}

// Функция для вычисления определителя матрицы последовательно
float determinant(float* matrix, int size) {
    if (size == 1) {
        return matrix[0];
    }

    if (size == 2) {
        return matrix[0] * matrix[3] - matrix[1] * matrix[2];
    }

    float det = 0;
    float sign = 1;

    for (int i = 0; i < size; i++) {
        // Используем динамический массив для минора
        float* minor = (float*)calloc((size - 1) * (size - 1), sizeof(float));

        for (int j = 1; j < size; j++) {
            for (int k = 0, col = 0; k < size; k++) {
                if (k != i) {
                    minor[(j - 1) * (size - 1) + col++] = matrix[j * size + k];
                }
            }
        }

        det += sign * matrix[i] * determinant(minor, size - 1);

        // Освобождение памяти, выделенной под минор
        free(minor);

        sign = -sign;
    }

    return det;
}

void fillMatrixRandom(float* matrix, int size) {

    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            // Генерация случайного числа в диапазоне от 0 до 1
            matrix[i * size + j] = (float)rand() / RAND_MAX;
        }
    }
}

void fillVectorRandom(float* vector, int size) {

    for (int i = 0; i < size; ++i) {
        vector[i] = rand() % 10;
    }
}

// Функция для сравнения на равенство матриц
bool compareMatrices(float* matrixA, float* matrixB, int size, float precision) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            // Функция fabs() для получения абсолютного значения разницы между элементами
            if (fabs(matrixA[i * size + j] - matrixB[i * size + j]) > precision) {
                return false;  // Если разница больше заданной точности, вернуть false
            }
        }
    }
    return true;  // Если все элементы совпадают с заданной точностью, вернуть true
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
    do {
        printf("Enter the size of the matrix A: ");
        if (scanf_s("%d", &MATRIX_SIZE) != 1 || MATRIX_SIZE <= 0) {
            printf("Invalid input. Please enter a positive integer.\n");
            while (getchar() != '\n');  // Очистка буфера ввода
        }
    } while (MATRIX_SIZE <= 0);

    // Динамическое выделение памяти для матрицы A и вектора B
    float* matrixA = (float*)calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));
    float* matrixB = (float*)calloc(MATRIX_SIZE, sizeof(float));

    // Ввод матрицы A
    //fillMatrixRandom(matrixA, MATRIX_SIZE);

    printf("Enter the matrix A (%dx%d):\n", MATRIX_SIZE, MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        for (int j = 0; j < MATRIX_SIZE; ++j) {
            printf("A[%d][%d]: ", i, j);
            while (scanf_s("%f", &matrixA[i * MATRIX_SIZE + j]) != 1) {
                printf("Invalid input. Please enter a valid floating-point number.\n");
                while (getchar() != '\n');
            }
        }
    }


    // Ввод вектора B
    //fillVectorRandom(matrixB, MATRIX_SIZE);

    printf("Enter the vector B (%d elements):\n", MATRIX_SIZE);
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        printf("B[%d]: ", i);

        // Проверка ввода на float
        while (scanf_s("%f", &matrixB[i]) != 1) {
            printf("Invalid input. Please enter a valid floating-point number.\n");
            // Очистка буфера ввода
            while (getchar() != '\n');
            printf("B[%d]: ", i);
        }
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
    float* matrixL = (float*)calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));
    float* matrixU = (float*)calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));

    cl_mem bufferL = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);
    cl_mem bufferU = clCreateBuffer(context, CL_MEM_READ_WRITE,
        sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, NULL, NULL);

    // Выполнение LU-разложения матрицы на GPU
    auto startGPU = std::chrono::high_resolution_clock::now();
    clSetKernelArg(kernelLU, 0, sizeof(cl_mem), &bufferA);
    clSetKernelArg(kernelLU, 1, sizeof(cl_mem), &bufferL);
    clSetKernelArg(kernelLU, 2, sizeof(cl_mem), &bufferU);
    clSetKernelArg(kernelLU, 3, sizeof(int), &matrixSize);

    size_t globalSizeLU[2] = { MATRIX_SIZE, MATRIX_SIZE };
    clEnqueueNDRangeKernel(queue, kernelLU, 2, NULL, globalSizeLU, NULL, 0, NULL, NULL);
    auto endGPU = std::chrono::high_resolution_clock::now();
    double GPUworkingTime = std::chrono::duration<double, std::milli>(endGPU - startGPU).count();

    clEnqueueReadBuffer(queue, bufferL, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, matrixL, 0, NULL, NULL);
    clEnqueueReadBuffer(queue, bufferU, CL_TRUE, 0, sizeof(float) * MATRIX_SIZE * MATRIX_SIZE, matrixU, 0, NULL, NULL);


    printMatrix("Matrix L", matrixL, MATRIX_SIZE, MATRIX_SIZE);
    printMatrix("Matrix U", matrixU, MATRIX_SIZE, MATRIX_SIZE);

    float* matrixResult = (float*)calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));
    //Умножение матриц
    auto startCPUmMultiply = std::chrono::high_resolution_clock::now();
    matrixMultiply(matrixL, matrixU, matrixResult, MATRIX_SIZE, MATRIX_SIZE);
    auto endCPUmMultiply = std::chrono::high_resolution_clock::now();
    double CPUParallelWorkingTimemMultiply = std::chrono::duration<double, std::milli>(endCPUmMultiply - startCPUmMultiply).count();

    printMatrix("Matrix Result (L * U)", matrixResult, MATRIX_SIZE, MATRIX_SIZE);
    std::cout << "CPU parallel matrix multiply time: " << CPUParallelWorkingTimemMultiply / 1000 << " milliseconds" << std::endl;
    bool matricesEqual = compareMatrices(matrixA, matrixResult, MATRIX_SIZE, 0.00001);

    if (matricesEqual) {
        printf("Matrices A and L*U are equal.\n");
    }
    else {
        printf("Matrices A and L*U are not equal.\n");
    }
    // Вычисление определителя
    auto startCPUdet = std::chrono::high_resolution_clock::now();
    long double det = determinantFromLU(matrixU, MATRIX_SIZE);
    auto endCPUdet = std::chrono::high_resolution_clock::now();
    //Последовательное вычисление определителя
    auto startdet = std::chrono::high_resolution_clock::now();
    float detP = determinant(matrixA, MATRIX_SIZE);
    auto enddet = std::chrono::high_resolution_clock::now();
    printf("Determinant: %f\n", det);
    double CPUParallelWorkingTimeDet = std::chrono::duration<double, std::milli>(endCPUdet - startCPUdet).count();
    double TimeDet = std::chrono::duration<double, std::milli>(enddet - startdet).count();
    std::cout << "CPU parallel calculating determinant time: " << CPUParallelWorkingTimeDet / 1000 << " milliseconds" << std::endl;
    std::cout << "Sequentially calculating determinant time: " << TimeDet / 1000 << " milliseconds" << std::endl;


    // Решение системы линейных уравнений Ax = B
    float* solution = (float*)calloc(MATRIX_SIZE, sizeof(float));
    double CPUParallelWorkingTime;

    // Проверка невырожденности матрицы A
    if (det != 0.0f) {
        printf("Matrix A is non-singular (det(A) != 0)\n");

        auto startCPU = std::chrono::high_resolution_clock::now();
        solveLinearSystem(matrixL, matrixU, matrixB, solution, MATRIX_SIZE);
        auto endCPU = std::chrono::high_resolution_clock::now();
        CPUParallelWorkingTime = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();

        printf("Solution of Ax = B:\n");
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            printf("%f\n", solution[i]);
        }
        std::cout << "CPU parallel solution of SoLE working time: " << CPUParallelWorkingTime / 1000 << " milliseconds" << std::endl;

    }
    else {
        printf("Matrix A is singular (det(A) = 0), cannot solve the system.\n");
    }

    std::cout << "GPU LU-decomposition working time: " << GPUworkingTime / 1000 << " milliseconds" << std::endl;

    // Освобождение памяти
    clReleaseMemObject(bufferL);
    clReleaseMemObject(bufferU);
    clReleaseKernel(kernelLU);
    clReleaseProgram(luDecompositionProgram);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);

    free(matrixA);
    free(matrixB);
    free(matrixL);
    free(matrixU);
    free(matrixResult);
    free(solution);

    return 0;
}
