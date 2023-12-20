#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <chrono>
#include <iostream>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <vector>

void luDecomposition(float* matrixA, float* matrixL, float* matrixU, int MATRIX_SIZE) {
#pragma omp parallel for
    for (int i = 0; i < MATRIX_SIZE; ++i) {
        // U матрица
#pragma omp for
        for (int k = i; k < MATRIX_SIZE; ++k) {
            float sum = 0.0f;
            for (int j = 0; j < i; ++j) {
                sum += matrixL[i * MATRIX_SIZE + j] * matrixU[j * MATRIX_SIZE + k];
            }
            matrixU[i * MATRIX_SIZE + k] = matrixA[i * MATRIX_SIZE + k] - sum;
        }

        // L матрица
#pragma omp for
        for (int k = i; k < MATRIX_SIZE; ++k) {
            if (i == k) {
                matrixL[i * MATRIX_SIZE + i] = 1.0f;
            }
            else {
                float sum = 0.0f;
                for (int j = 0; j < i; ++j) {
                    sum += matrixL[k * MATRIX_SIZE + j] * matrixU[j * MATRIX_SIZE + i];
                }
                matrixL[k * MATRIX_SIZE + i] = (matrixA[k * MATRIX_SIZE + i] - sum) / matrixU[i * MATRIX_SIZE + i];
            }
        }
    }
}


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

    const char* graphFile = "C:/projects/avs/results/cpu_results.csv";

    std::vector<int> matrixSizes = { 10, 20, 50, 100, 200, 300 };

    for (int MatrixSize : matrixSizes)
    {

        // Ввод размера матрицы
        int MATRIX_SIZE = MatrixSize;
        //do {
        //    printf("Enter the size of the matrix A: ");
        //    if (scanf_s("%d", &MATRIX_SIZE) != 1 || MATRIX_SIZE <= 0) {
        //        printf("Invalid input. Please enter a positive integer.\n");
        //        while (getchar() != '\n');  // Очистка буфера ввода
        //    }
        //} while (MATRIX_SIZE <= 0);

        // Динамическое выделение памяти для матрицы A и вектора B
        float* matrixA = (float*)calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));
        float* matrixB = (float*)calloc(MATRIX_SIZE, sizeof(float));

        // Ввод матрицы A
        fillMatrixRandom(matrixA, MATRIX_SIZE);

        /*printf("Enter the matrix A (%dx%d):\n", MATRIX_SIZE, MATRIX_SIZE);
        for (int i = 0; i < MATRIX_SIZE; ++i) {
            for (int j = 0; j < MATRIX_SIZE; ++j) {
                printf("A[%d][%d]: ", i, j);
                while (scanf_s("%f", &matrixA[i * MATRIX_SIZE + j]) != 1) {
                    printf("Invalid input. Please enter a valid floating-point number.\n");
                    while (getchar() != '\n');
                }
            }
        }*/


        // Ввод вектора B
        fillVectorRandom(matrixB, MATRIX_SIZE);

        //printf("Enter the vector B (%d elements):\n", MATRIX_SIZE);
        //for (int i = 0; i < MATRIX_SIZE; ++i) {
        //    printf("B[%d]: ", i);

        //    // Проверка ввода на float
        //    while (scanf_s("%f", &matrixB[i]) != 1) {
        //        printf("Invalid input. Please enter a valid floating-point number.\n");
        //        // Очистка буфера ввода
        //        while (getchar() != '\n');
        //        printf("B[%d]: ", i);
        //    }
        //}

        //printMatrix("Matrix A", matrixA, MATRIX_SIZE, MATRIX_SIZE);
        int matrixSize = MATRIX_SIZE;

        // Выделение памяти для L и U матриц
        float* matrixL = (float*)calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));
        float* matrixU = (float*)calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));

        auto startCPU = std::chrono::high_resolution_clock::now(); //Начало отсчета

        // Выполнение LU-разложения матрицы
        luDecomposition(matrixA, matrixL, matrixU, MATRIX_SIZE);

        //printMatrix("Matrix L", matrixL, MATRIX_SIZE, MATRIX_SIZE);
        //printMatrix("Matrix U", matrixU, MATRIX_SIZE, MATRIX_SIZE);

        float* matrixResult = (float*)calloc(MATRIX_SIZE * MATRIX_SIZE, sizeof(float));
        //Умножение матриц
        matrixMultiply(matrixL, matrixU, matrixResult, MATRIX_SIZE, MATRIX_SIZE);

        //printMatrix("Matrix Result (L * U)", matrixResult, MATRIX_SIZE, MATRIX_SIZE);
        bool matricesEqual = compareMatrices(matrixA, matrixResult, MATRIX_SIZE, 0.00001);

        if (matricesEqual) {
            printf("Matrices A and L*U are equal.\n");
        }
        else {
            printf("Matrices A and L*U are not equal.\n");
        }
        // Вычисление определителя
        long double det = determinantFromLU(matrixU, MATRIX_SIZE);
        printf("Determinant: %f\n", det);


        // Решение системы линейных уравнений Ax = B
        float* solution = (float*)calloc(MATRIX_SIZE, sizeof(float));

        // Проверка невырожденности матрицы A
        if (det != 0.0f) {
            printf("Matrix A is non-singular (det(A) != 0)\n");

            solveLinearSystem(matrixL, matrixU, matrixB, solution, MATRIX_SIZE);

            printf("Solution of Ax = B:\n");
            for (int i = 0; i < MATRIX_SIZE; ++i) {
                printf("%f\n", solution[i]);
            }

        }
        else {
            printf("Matrix A is singular (det(A) = 0), cannot solve the system.\n");
        }

        auto endCPU = std::chrono::high_resolution_clock::now();  // Конец отсчета
        double CPUworkingTime = std::chrono::duration<double, std::milli>(endCPU - startCPU).count();
        std::cout << "CPU time: " << CPUworkingTime << " milliseconds" << std::endl;

        std::cout << std::setw(20) << MatrixSize << " | " << std::setw(20) << std::fixed << std::setprecision(3)
            << CPUworkingTime << "\n";
        //<< " | " << std::setw(20) << minVal << "\n";

        std::ofstream outFile(graphFile, std::ios::app);

        outFile << std::fixed << std::setprecision(3) << CPUworkingTime << ",";
        outFile.close();

        free(matrixA);
        free(matrixB);
        free(matrixL);
        free(matrixU);
        free(matrixResult);
        free(solution);
    }


    std::ofstream outFile(graphFile, std::ios::app);

    if (!outFile.is_open()) {
        std::cerr << "Unable to open the file: " << graphFile << std::endl;
        return 1;
    }
    outFile << '\n';
    outFile.close();


    return 0;
}
