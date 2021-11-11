#ifndef GPR_H
#define GPR_H

#pragma once
#include "Matrix_omp.hpp"
using std::vector;

double kernel(double x1, double y1, double x2, double y2, double l1, double l2);
double fxy(double x1, double y1, double x2, double y2, double l1, double l2, double c1, double c2);
double gxy(double x, double y, double x0, double y0, double gamma);
vector<vector<int> > train_test_split(int n, double ratio);
Matrix<double> init_grid_search_space(const double l1[], const double l2[]);
Matrix<double> init_grid(int grid_size);
vector<double> init_f(Matrix<double> &grid, vector<vector<int> > &train_test_idx, const int func);
Matrix<double> init_K(Matrix<double> &grid, double l1, double l2, vector<vector<int> > &train_test_idx);
vector<double> init_k(Matrix<double> &grid, vector<vector<int> > &train_test_idx, vector<double> &rstar, double l1, double l2);
Matrix<double> init_A(double t, Matrix<double> &K);
vector<double> forward_substitution(Matrix<double> &L, vector<double> &f);
vector<double> backward_substitution(Matrix<double> &U, vector<double> &y);
double solve_LEs(Matrix<double> &L, Matrix<double> &LT, vector<double> &f, vector<double> &k);
double dot_product(vector<double> &v1, vector<double> &v2);

template<typename T>
void vec_display(vector<T> &v)
{
    int n = v.size();
    for (int i = 0; i < n; i++)
    {
        cout << v[i] << ", ";
    }
    cout << endl;
}

#endif