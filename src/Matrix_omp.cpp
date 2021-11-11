#include <string>
#include <iostream>
#include <utility>
#include "Matrix_omp.hpp"
#include <omp.h>
using std::cout;
using std::endl;

// Pararmeter Constructor
template<typename T>
Matrix<T>::Matrix(int rows, int cols, const T &initial)
{
    mat.resize(rows);
    for (int i = 0; i < mat.size(); i++)
    {
        mat[i].resize(cols, initial);
    }
    _rows = rows;
    _cols = cols;
}

// Copy Constructor
template<typename T>
Matrix<T>::Matrix(const Matrix<T> &obj)
{
    mat = obj.mat;
    _rows = obj.get_rows();
    _cols = obj.get_cols();
}

// Virtual Destructor
/* 
Avoid undefined behavior when deleting a derived class object 
using a pointer of base class type that has a non-virtual destructor.
*/
template<typename T>
Matrix<T>::~Matrix() {}

// Assignment Operator
template<typename T>
Matrix<T> &Matrix<T>::operator=(const Matrix<T> &obj)
{
    if (&obj == this)
        return *this;

    int new_rows = obj.get_rows();
    int new_cols = obj.get_cols();

    mat.resize(new_rows);
    //# pragma omp parallel for
    for (int i = 0; i < new_rows; i++)
    {
        mat[i].resize(new_cols);
    }

    #pragma omp parallel for
    for (int i = 0; i < new_rows; i++)
    {
        for (int j = 0; j < new_cols; j++)
        {
            mat[i][j] = obj(i, j);
        }
    }
    _rows = new_rows;
    _cols = new_cols;

    return *this;
}

// Addition of this matrix and another
template<typename T>
Matrix<T> Matrix<T>::operator+(Matrix<T> &obj)
{
    Matrix<T> res(_rows, _cols, 0);

    #pragma omp parallel for
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            res(i, j) = this->mat[i][j] + obj(i, j);
        }
    }
    return res;
}

// Cumulative addition of this matrix and another
template<typename T>
Matrix<T>& Matrix<T>::operator+=(Matrix<T> &obj)
{
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            this->mat[i][j] += obj(i, j);
        }
    }
    return *this;
}

// Subtraction of this matrix and another
template<typename T>
Matrix<T> Matrix<T>::operator-(Matrix<T> &obj)
{
    Matrix<T> res(_rows, _cols, 0);

    #pragma omp parallel for
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            res(i, j) = this->mat[i][j] - obj(i, j);
        }
    }
    return res;
}

// Cumulative subtraction of this matrix and another
template<typename T>
Matrix<T>& Matrix<T>::operator-=(Matrix<T> &obj)
{
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            this->mat[i][j] -= obj(i, j);
        }
    }
    return *this;
}

// Left multiplication of this matrix and another
template<typename T>
Matrix<T> Matrix<T>::operator*(Matrix<T> &obj)
{
    if (_cols == obj.get_rows())
    {
        int cols = obj.get_cols();
        double temp;
        Matrix res(_rows, cols, 0);
        Matrix objT = obj.transpose();

        #pragma omp parallel for
        for (int i = 0; i < _rows; i++)
        {
            for (int j = 0; j < cols; j++)
            {
                double temp = 0;
                for (int k = 0; k < cols; k++)
                {
                    temp += this->mat[i][k] * objT(j, k);
                }
                res(i, j) = temp;
            }
        }
        return res;
    }
    else
    {
        throw "Error: Dimensions do not match!";
    }
}

// Cumulative left multiplication of this matrix and another
template<typename T>
Matrix<T>& Matrix<T>::operator*=(Matrix<T> &obj)
{
    (*this) = (*this) * obj;
    return *this;
}

// Transpose of this matrix
template <typename T>
Matrix<T> Matrix<T>::transpose()
{
    Matrix res(_rows, _cols, 0);

    #pragma omp parallel for
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            res(i, j) = this->mat[j][i];
        }
    }
    return res;
}

// Find determinant of this matrix
template<typename T>
double Matrix<T>::det()
{
    double v_diag, v_next, res = 1, factor = 1;
    int idx_r;

    // Array for storing row elements.
    T temp[_rows + 1];

    // Make a working copy of this matrix.
    Matrix<T> mat_copy(*this);

    // Traverse the diagonal elements
    for (int i = 0; i < _cols; i++)
    {
        idx_r = i; // Initialize row index

        // Find the index at which the value is non-zero
        while (idx_r < _rows && mat_copy(idx_r, i) == 0)
        {
            idx_r++;
        }

        // If there is no non-zero element, the determinant is zero.
        if (idx_r == _rows)
            return 0;

        // If there is a non-zero element in row idx_r,
        // swap diagnoal element row i and row idx_r.
        // Change the sign of det after the row exchange.
        if (idx_r != i)
        {
            for (int j = 0; j < _cols; j++)
            {
                std::swap(mat_copy(idx_r, j), mat_copy(i, j));
            }
            res = -res;
        }

        // Store the values of in the row where the diagnoal element is.
        for (int j = 0; j < _cols; j++)
        {
            temp[j] = mat_copy(i, j);
        }

        // Traverse every row below the diagonal element row.
        for (int j = i + 1; j < _rows; j++)
        {
            v_diag = temp[i];        // Value of diagonal element
            v_next = mat_copy(j, i); // Value of element in the next row

            for (int k = 0; k < _cols; k++)
            {
                mat_copy(j, k) = v_diag * mat_copy(j, k) - v_next * temp[k];
            }
            factor *= v_diag; // det(kA) = k * det(A)
        }
    }

    // Multiply the diagonal elements to get determinant
    for (int i = 0; i < _rows; i++)
    {
        res = res * mat_copy(i, i);
    }
    return res / factor; // det(A) = det(kA) / k
}

// Inverse this matrix
template<typename T>
Matrix<T> Matrix<T>::inverse()
{
    Matrix<T> mat_aug(_rows, 2 * _cols, 0);
    Matrix<T> mat_inv(_rows, _cols, 0);
    double ratio;

    // Append the identity matrix (_rows, _cols) at the end of this matrix
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            mat_aug(i, j) = this->mat[i][j];
        }
    }
    for (int i = 0; i < _rows; i++)
    {
        for (int j = _cols; j < 2 * _cols; j++)
        {
            if (j == i + _cols)
            {
                mat_aug(i, j) = 1;
            }
        }
    }

    // Swap the rows of matrix
    for (int i = _rows - 1; i > 0; i--)
    {
        if (mat_aug(i - 1, 0) < mat_aug(i, 0))
        {
            // Swap the (i - 1)th and ith row
            mat_aug.mat[i - 1].swap(mat_aug.mat[i]);
        }
    }

    // Update a row by the sum of itself and a constant multiple of another row
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            if (j != i)
            {
                ratio = mat_aug(j, i) / mat_aug(i, i);
                for (int k = 0; k < 2 * _cols; k++)
                {
                    mat_aug(j, k) -= mat_aug(i, k) * ratio;
                }
            }
        }
    }

    // Divide row elements by the diagonal element in the row to make
    // the principal diagnonal elements equal to 1.
    for (int i = 0; i < _rows; i++)
    {
        for (int j = _cols; j < 2 * _cols; j++)
        {
            mat_aug(i, j) /= mat_aug(i, i);
        }
    }

    // Extract the inversed matrix.
    int p = 0;
    for (int i = 0; i < _rows; i++)
    {
        int q = 0;
        for (int j = _cols; j < 2 * _cols; j++)
        {
            mat_inv(p, q) = mat_aug(i, j);
            q++;
        }
        p++;
    }
    return mat_inv;
}

template<typename T>
Matrix<T> Matrix<T>::cholesky_decomposition()
{
    Matrix<T> res(_rows, _rows, 0);

    for (int j = 0; j < _rows; j++)
    {
        double sum = 0;
        
        for (int k = 0; k < j; k++)
        {
            sum += res(j, k) * res(j, k);
        }
        res(j, j) = sqrt(this->mat[j][j] - sum);

        #pragma omp parallel for private(sum)
        for (int i = j + 1; i < _rows; i++)
        {
            sum = 0;
            for (int k = 0; k < j; k++)
            {
                sum += res(i, k) * res(j, k);
            }
            res(i, j) = (this->mat[i][j] - sum) / res(j, j);
        }
    }
    
   /*
    #pragma omp parallel for
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j <= i; j++)
        {
            double sum = 0;
            for (int k = 0; k < j; k++)
                sum += res(i, k) * res(j, k);

            if (i == j)
                res(i, j) = sqrt(this->mat[i][j] - sum);
            else
                res(i, j) = (this->mat[i][j] - sum) / res(j, j);
        }
    }
    */
    return res;
}

// Matrix/scalar addition
template<typename T>
Matrix<T> Matrix<T>::operator+(const T &obj)
{
    Matrix<T> res(_rows, _cols, 0);

    #pragma omp parallel for
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            res(i, j) = this->mat[i][j] + obj;
        }
    }
    return res;
}

// Matrix/scalar subtraction
template<typename T>
Matrix<T> Matrix<T>::operator-(const T &obj)
{
    Matrix<T> res(_rows, _cols, 0);

    #pragma omp parallel for
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            res(i, j) = this->mat[i][j] - obj;
        }
    }
    return res;
}

// Matrix/scalar multiplication
template<typename T>
Matrix<T> Matrix<T>::operator*(const T &obj)
{
    Matrix res(_rows, _cols, 0);

    #pragma omp parallel for
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            res(i, j) = this->mat[i][j] * obj;
        }
    }
    return res;
}

// Matrix/scalar division
template<typename T>
Matrix<T> Matrix<T>::operator/(const T &obj)
{
    Matrix res(_rows, _cols, 0);

    #pragma omp parallel for
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            res(i, j) = this->mat[i][j] / obj;
        }
    }
    return res;
}

// Multiplication of this matrix and a vector
template<typename T>
std::vector<T> Matrix<T>::operator*(const std::vector<T> &obj)
{
    std::vector<T> res(_cols, 0);

    #pragma omp parallel for
    for (int i = 0; i < _rows; i++)
    {
        double temp = 0;
        for (int j = 0; j < _cols; j++)
        {
            temp += this->mat[i][j] * obj[j]; 
        }
        res[i] = temp;
    }
    return res;
}

// Access individual elements
template<typename T>
T &Matrix<T>::operator()(const int &row, const int &col)
{
    return this->mat[row][col];
}

// Access individual elements (const)
template<typename T>
const T &Matrix<T>::operator()(const int &row, const int &col) const
{
    return this->mat[row][col];
}

// Get the number of rows of this matrix
template<typename T>
int Matrix<T>::get_rows() const
{
    return this->_rows;
}

// Get the number of cols of this matrix
template <typename T>
int Matrix<T>::get_cols() const
{
    return this->_cols;
}

// Display matrix
template<typename T>
void Matrix<T>::display()
{
    for (int i = 0; i < _rows; i++)
    {
        for (int j = 0; j < _cols; j++)
        {
            cout << this->mat[i][j] << ", ";
        }
        cout << endl;
    }
}
