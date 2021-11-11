//#include <iostream>
#define _USE_MATH_DEFINES 

#include <omp.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstddef>
#include <algorithm>
#include <valarray>
#include <time.h>
#include <random>
#include "Matrix_omp.hpp"
#include "gpr.hpp"
using std::cout;
using std::endl;
using std::vector;

double kernel(double x1, double y1, double x2, double y2, double l1, double l2)
{
    double res = (1.0 / sqrt(2*M_PI)) * exp(-0.5*(pow((x1 - x2) / l1, 2) + pow((y1 - y2) / l2, 2)));
    return res;
}

double fxy(double x1, double y1, double x2, double y2, double l1, double l2, double c1, double c2)
{
    double res = kernel(x1, y1, x2, y2, l1, l2) + c1 * x1 + c2 * y1;
    return res;
}

double gxy(double x, double y, double x0, double y0, double gamma)
{
    double res = (1.0 / (2*M_PI)) * (gamma / (pow(pow(x - x0, 2) + pow(y - y0, 2) + pow(gamma, 2), 1.5)));
    return res;
}

vector<vector<int> > train_test_split(int n, double ratio)
{
    /* 
    Return a vector (train_idxs, test_idxs), in which the two elements 
    are vectors of train index and test index, respectively.
    */
    vector<int> idx(n, 0);
    int n_test = round(ratio * n);
    int n_train = n - n_test;
    int seed = 111;
    vector<vector<int> > res;

    #pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        idx[i] = i;
    }

    std::shuffle(idx.begin(), idx.end(), std::default_random_engine(seed));
    vector<int> train_idxs(idx.begin(), idx.begin() + n_train);
    vector<int> test_idxs(idx.begin() + n_train, idx.end());

    res.push_back(train_idxs);
    res.push_back(test_idxs);

    return res;
}

Matrix<double> init_grid_search_space(const double l1[], const double l2[])
{
    int n1 = (int)(floor((l1[1] - l1[0]) / l1[2]) + 1);
    int n2 = (int)(floor((l2[1] - l2[0]) / l2[2]) + 1);

    Matrix<double> res(n1 * n2, 2, 0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < n1; i++)
    {
        for (int j = 0; j < n2; j++)
        {
            res(i * n2 + j, 0) = l1[2] * i + l1[0];
            res(i * n2 + j, 1) = l2[2] * j + l2[0];
        }
    }
    return res;
}

Matrix<double> init_grid(int grid_size)
{
    int n = grid_size * grid_size;
    double h = 1 / ((double)grid_size + 1);
    Matrix<double> res(n, 2, 0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < grid_size; i++)
    {
        for (int j = 0; j < grid_size; j++)
        {
            res(i * grid_size + j, 0) = (i + 1) * h;
            res(i * grid_size + j, 1) = (j + 1) * h;
        }
    }

    return res;
}

vector<double> init_f(Matrix<double> &grid, vector<vector<int> > &train_test_idx, const int func)
{
    //srand(111);
    srand(time(NULL));
    int m = round(sqrt(grid.get_rows())); // Dimension of the data points grid
    int train_space = train_test_idx[0].size(); // Number of training points
    vector<double> res(train_space, 0);

    if (func == 0)
    {
        #pragma omp parallel for
        for (int i = 0; i < train_space; i++)
        {
            double f_rand = 0.02 * ((double)rand() / RAND_MAX - 0.5);
            int j = train_test_idx[0][i];
            res[i] = f_rand + fxy(grid(j, 0), grid(j ,1), 0.25, 0.25, 2.0 / m, 2.0 / m, 0.2, 0.1);
        }
    }
    else
    {
        #pragma omp parallel for
        for (int i = 0; i < train_space; i++)
        {
            double g_rand = 0.02 * ((double)rand() / RAND_MAX - 0.5);
            int j = train_test_idx[0][i];
            res[i] = g_rand + gxy(grid(j, 0), grid(j ,1), 0.5, 0.5, 0.2);
        }
    }

    return res;
}

Matrix<double> init_K(Matrix<double> &grid, double l1, double l2, vector<vector<int> > &train_test_idx)
{
    int train_space = train_test_idx[0].size(); // Number of training points
    Matrix<double> res(train_space, train_space, 0);

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < train_space; i++)
    {
        for (int j = 0; j < train_space; j++)
        {
            int p = train_test_idx[0][i];
            int q = train_test_idx[0][j];
            res(i, j) = kernel(grid(p, 0), grid(p, 1), grid(q, 0), grid(q, 1), l1, l2);
        }
    }

    return res;
}

vector<double> init_k(Matrix<double> &grid, vector<vector<int> > &train_test_idx, vector<double> &rstar, double l1, double l2)
{
    int train_space = train_test_idx[0].size(); // Number of training points
    vector<double> res(train_space, 0);

    #pragma omp parallel for
    for (int i = 0; i < train_space; i++)
    {
        int j = train_test_idx[0][i];
        res[i] = kernel(grid(j, 0), grid(j, 1), rstar[0], rstar[1], l1, l2);
    }

    return res;
}

Matrix<double> init_A(double t, Matrix<double> &K)
{
    int n = K.get_rows();
    Matrix<double> I(n, n, 0);
    Matrix<double> res(n, n, 0);
    
    //#pragma omp parallel for
    for (int i = 0; i < n; i++)
    {
        I(i, i) = 1;
    }
    
    res = I * t + K;
    return res;
}

std::vector<double> forward_substitution(Matrix<double> &L, std::vector<double> &f)
{
    int n = L.get_rows();
    std::vector<double> res(n, 0);

    res[0] = f[0] / L(0, 0);
    for (int i = 1; i < n; i++)
    {
        double sum = 0;
        
        #pragma omp parallel for reduction(+: sum)
        for (int j = 0; j < i; j++)
        {
            sum += L(i, j) * res[j];
        }
        res[i] = (f[i] - sum) / L(i, i);
    }
    return res;
}

std::vector<double> backward_substitution(Matrix<double> &U, std::vector<double> &y)
{
    int n = U.get_rows();
    std::vector<double> res(n, 0);

    res[n - 1] = y[n - 1] / U(n - 1, n - 1);
    for (int i = n - 1; i >= 0; i--)
    {
        double sum = 0;
        
        #pragma omp parallel for reduction(+: sum)
        for (int j = i + 1; j < n; j++)
        {
            sum += U(i, j) * res[j];
        }
        res[i] = (y[i] - sum) / U(i, i);
    }
    return res;
}

double solve_LEs(Matrix<double> &L, Matrix<double> &LT, std::vector<double> &f, std::vector<double> &k)
{
    int n = L.get_rows();
    std::vector<double> y(n, 0);
    std::vector<double> x(n, 0);
    double res;

    y = forward_substitution(L, f);
    x = backward_substitution(LT, y);
    res = dot_product(k, x);
    
    return res;
}

double dot_product(std::vector<double> &v1, std::vector<double> &v2)
{
    if (v1.size() != v2.size())
    {
        throw "Vector sizes are not the same!";
    }
    else
    {
        double res = 0;
        int n = v1.size();
        
        #pragma omp parallel for reduction(+: res)
        for (int i = 0; i < n; i++)
        {
            res += v1[i] * v2[i];
        }
        return res;
    }
}

