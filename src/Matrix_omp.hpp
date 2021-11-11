#ifndef MATRIX_OMP
#define MATRIX_OMP

#include <vector>

template <typename T>
class Matrix
{
private:
    std::vector<std::vector<T> > mat;
    int _rows;
    int _cols;

public:
    //Matrix();
    Matrix(int _rows, int _cols, const T &_initial);
    Matrix(const Matrix<T> &obj);
    virtual ~Matrix();

    // Assigment operator
    Matrix<T> &operator=(const Matrix<T> &obj);

    // Matrix mathematical operations
    Matrix<T> operator+(Matrix<T> &obj);
    Matrix<T>& operator+=(Matrix<T> &obj);
    Matrix<T> operator-(Matrix<T> &obj);
    Matrix<T>& operator-=(Matrix<T> &obj);
    Matrix<T> operator*(Matrix<T> &obj);
    Matrix<T>& operator*=(Matrix<T> &obj);
    Matrix<T> transpose();
    double det();
    Matrix<T> inverse();
    Matrix<T> cholesky_decomposition();

    // Matrix/scalar operations
    Matrix<T> operator+(const T &obj);
    Matrix<T> operator-(const T &obj);
    Matrix<T> operator*(const T &obj);
    Matrix<T> operator/(const T &obj);

    // Matrix/vector operations
    std::vector<T> operator*(const std::vector<T> &obj);

    // Access the individual elements
    T &operator()(const int &row, const int &col);
    const T &operator()(const int &row, const int &col) const;

    // Access the row and column sizes
    int get_rows() const;
    int get_cols() const;

    // Utilities
    void display();
};

#include "Matrix_omp.cpp"

#endif