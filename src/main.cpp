#include <iostream>
#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <limits>
#include <time.h>
#include "Matrix_omp.hpp"
#include "gpr.hpp"
using std::cout;
using std::endl;

int main(int argc, char** argv)
{
    // Enter a data point (x, y) to check the predicted value of 
    // f(x, y) using optimal hyperparameters l1 and l2
    std::vector<double> r_validate(2, 0); 
    int grid_size = std::atoi(argv[1]); // 2D unit square (grid_size * grid_size)
    r_validate[0] = std::stod(argv[2]);
    r_validate[1] = std::stod(argv[3]);
    int nthreads = std::atoi(argv[4]); // Number of OpenMP threads to use
    const int func_tag = std::atoi(argv[5]);
    
    int m = pow(grid_size, 2);
    Matrix<double> grid(grid_size, grid_size, 0);
    std::vector<double> f; // Observed data points f(x, y)
    const double t = 0.01; // Hyperparameter t
    double fstar; // Predicted value of f(x, y)
    double start, end; // For timing purpose

    // Define the ranges of l1 and l2 with (start, end, step)
    const double l1[3] = {0.01, 0.41, 0.02};
    const double l2[3] = {0.01, 0.41, 0.02};

    omp_set_dynamic(0);
    omp_set_num_threads(nthreads);

    #pragma omp parallel
    {
        #pragma omp single
        {
            int n = omp_get_num_threads();
            cout << "Actual #threads: " << n << endl;
        }
    }

    // Shuffle the index of all the observed data points and split into training and test set.
    // Train - 90%, Test - 10%  
    vector<vector<int> > train_test_idxs(train_test_split(m, 0.1));
    
    // Initialize the hyperparameter space as a (n, 2) matrix, in which 
    // each row is a combination of (l1, l2). 
    Matrix<double> hp_space(init_grid_search_space(l1, l2));
    int gs_size = hp_space.get_rows(); // Size of grid search

    // Initialize the 2D grid of data points
    grid = init_grid(grid_size);

    // Initialize training data points
    f = init_f(grid, train_test_idxs, func_tag); 

    double *mse_group;
    mse_group = new double [gs_size]; // Store the MSE for every hyperparameter pair
    int test_space = train_test_idxs[1].size(); // Load the number of test data points
    
    // Do grid search
    start = omp_get_wtime(); // Start of timing
    #pragma omp parallel for
    for (int i = 0; i < gs_size; i++)
    {
        double local_mse = 0;
        Matrix<double> K(m, m, 0);
        K = init_K(grid, hp_space(i, 0), hp_space(i, 1), train_test_idxs);

        #pragma omp parallel for reduction(+: local_mse)
        for (int j = 0; j < test_space; j++)
        {
            
            Matrix<double> A(m, m, 0);
            Matrix<double> L(m, m, 0);
            Matrix<double> LT(m, m, 0);
            
            std::vector<double> k;
            std::vector<double> z;
            int test_i = train_test_idxs[1][j]; // Get index of the jth test data point
            vector<double> rstar(2, 0);
            rstar[0] = grid(test_i, 0);
            rstar[1] = grid(test_i, 1);

            k = init_k(grid, train_test_idxs, rstar, hp_space(i, 0), hp_space(i, 1));
            A = init_A(t, K);
            L = A.cholesky_decomposition();
            LT = L.transpose();
            fstar = solve_LEs(L, LT, f, k);
            if (func_tag == 0)
            {
                local_mse += pow(fstar - fxy(rstar[0], rstar[1], 0.25, 0.25, \
                                2.0 / (double)grid_size, 2.0 / (double)grid_size, 0.2, 0.1), 2);
            }
            else
            {
                local_mse += pow(fstar - gxy(rstar[0], rstar[1], 0.5, 0.5, 0.2), 2);
            }   
        }
        local_mse /= (double) test_space;
        mse_group[i] = local_mse;
    }
    end = omp_get_wtime(); // End of timing

    double mse_min = std::numeric_limits<double>::infinity();
    int idx_min_mse;
    for (int i = 0; i < gs_size; i++)
    {
        if (mse_group[i] < mse_min)
        {
            mse_min = mse_group[i];
            idx_min_mse = i;
        }
    }
    delete[] mse_group;
    printf("Running Time: %.4f (s)\n", end - start);
    printf("Optimal hyperparameter pairs (l1, l2) = (%.2f, %.2f)\n", hp_space(idx_min_mse, 0), hp_space(idx_min_mse, 1));
    printf("Minimum MSE: %1.3e\n", mse_min);

    // Check the model with user input (x, y) and optimal (l1, l2)
    Matrix<double> K(m, m, 0);
    Matrix<double> A(m, m, 0);
    Matrix<double> L(m, m, 0);
    Matrix<double> LT(m, m, 0);
    double f_true;
    std::vector<double> k;
    std::vector<double> z;

    train_test_idxs = train_test_split(m, 0);
    f = init_f(grid, train_test_idxs, func_tag);
    K = init_K(grid, hp_space(idx_min_mse, 0), hp_space(idx_min_mse, 1), train_test_idxs);
    k = init_k(grid, train_test_idxs, r_validate, hp_space(idx_min_mse, 0), hp_space(idx_min_mse, 1));
    A = init_A(t, K);
    L = A.cholesky_decomposition();
    LT = L.transpose();
    fstar = solve_LEs(L, LT, f, k);
    if (func_tag == 0)
    {
        f_true = fxy(r_validate[0], r_validate[1], 0.25, 0.25, \
                2.0 / (double)grid_size, 2.0 / (double)grid_size, 0.2, 0.1);
    }
    else
    {
        f_true = gxy(r_validate[0], r_validate[1], 0.5, 0.5, 0.2);
    }
    printf("Quick validation with point (%.2f, %.2f)\n", r_validate[0], r_validate[1]);
    printf("f_star = %f\n", fstar);
    printf("f_true = %f\n", f_true);
    printf("Relative error: %.2f %%\n", fabs(fstar - f_true) / f_true * 100);
    return 0;
}