#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "Types.hpp"

template <MatrixUnit DataType = double>
class Matrix
{
 private:
  size_t rows_;
  size_t cols_;
  std::vector<DataType> data_;

  constexpr static DataType EPSILON =
      static_cast<double>(std::numeric_limits<DataType>::epsilon()) * 100.0;

  template <MatrixUnit U>
  friend class Matrix;

 public:
  Matrix(size_t rows, size_t cols, DataType initial_value = DataType(0))
      : rows_(rows), cols_(cols), data_(rows * cols, initial_value)
  {
    if (rows == 0 || cols == 0)
    {
      throw std::invalid_argument("Matrix dimensions must be positive");
    }
  }

  ~Matrix() = default;
  Matrix(const Matrix&) = default;
  Matrix& operator=(const Matrix&) = default;
  Matrix(Matrix&&) noexcept = default;
  Matrix& operator=(Matrix&&) noexcept = default;

  size_t Rows() const { return rows_; }
  size_t Cols() const { return cols_; }

  DataType& operator()(size_t row, size_t col) { return data_[row * cols_ + col]; }

  const DataType& operator()(size_t row, size_t col) const
  {
    return data_[row * cols_ + col];
  }

  DataType& At(size_t row, size_t col)
  {
    if (row >= rows_ || col >= cols_)
    {
      throw std::out_of_range("Matrix index out of range");
    }
    return data_[row * cols_ + col];
  }

  /** 转置 */
  Matrix Transpose() const
  {
    Matrix result(cols_, rows_);
    for (size_t i = 0; i < rows_; ++i)
    {
      for (size_t j = 0; j < cols_; ++j)
      {
        result(j, i) = (*this)(i, j);
      }
    }
    return result;
  }

  /** 迹 */
  DataType Trace() const
  {
    if (rows_ != cols_)
    {
      throw std::invalid_argument("Trace is only defined for square matrices");
    }
    DataType sum = DataType(0);
    for (size_t i = 0; i < rows_; ++i)
    {
      sum += (*this)(i, i);
    }
    return sum;
  }

  /** Frobenius 范数 */
  double Norm() const
  {
    double sum = 0.0;
    for (const auto& v : data_)
    {
      sum += static_cast<double>(v) * static_cast<double>(v);
    }
    return std::sqrt(sum);
  }

  template <MatrixUnit OtherType>
    requires Addable<DataType, OtherType, DataType>
  Matrix<DataType> operator+(const Matrix<OtherType>& other) const
  {
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
      throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    Matrix<DataType> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i)
    {
      for (size_t j = 0; j < cols_; ++j)
      {
        result(i, j) = (*this)(i, j) + static_cast<DataType>(other(i, j));
      }
    }
    return result;
  }

  template <MatrixUnit OtherType>
    requires Addable<DataType, OtherType, DataType>
  Matrix<DataType>& operator+=(const Matrix<OtherType>& other)
  {
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
      throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    for (size_t i = 0; i < rows_; ++i)
    {
      for (size_t j = 0; j < cols_; ++j)
      {
        (*this)(i, j) += static_cast<DataType>(other(i, j));
      }
    }
    return *this;
  }

  template <MatrixUnit OtherType>
    requires Addable<DataType, OtherType, DataType>
  friend Matrix<DataType> operator+(Matrix<DataType>&& lhs, const Matrix<OtherType>& rhs)
  {
    lhs += rhs;
    return std::move(lhs);
  }

  template <MatrixUnit OtherType>
    requires Subtractable<DataType, OtherType, DataType>
  Matrix<DataType> operator-(const Matrix<OtherType>& other) const
  {
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
      throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    Matrix<DataType> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i)
    {
      for (size_t j = 0; j < cols_; ++j)
      {
        result(i, j) = (*this)(i, j) - static_cast<DataType>(other(i, j));
      }
    }
    return result;
  }

  template <MatrixUnit OtherType>
    requires Subtractable<DataType, OtherType, DataType>
  Matrix<DataType>& operator-=(const Matrix<OtherType>& other)
  {
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
      throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    for (size_t i = 0; i < rows_; ++i)
    {
      for (size_t j = 0; j < cols_; ++j)
      {
        (*this)(i, j) -= static_cast<DataType>(other(i, j));
      }
    }
    return *this;
  }

  template <MatrixUnit OtherType>
    requires Multipliable<DataType, OtherType, DataType>
  Matrix<DataType> operator*(const Matrix<OtherType>& other) const
  {
    if (cols_ != other.rows_)
    {
      throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    Matrix<DataType> result(rows_, other.cols_, DataType(0));
    for (size_t i = 0; i < rows_; ++i)
    {
      for (size_t j = 0; j < cols_; ++j)
      {
        for (size_t k = 0; k < other.cols_; ++k)
        {
          result(i, k) += (*this)(i, j) * static_cast<DataType>(other(j, k));
        }
      }
    }
    return result;
  }

  /** 标量右乘 */
  Matrix<DataType> operator*(DataType scalar) const
  {
    Matrix<DataType> result(*this);
    for (auto& v : result.data_)
    {
      v *= scalar;
    }
    return result;
  }

  Matrix<DataType>& operator*=(DataType scalar)
  {
    for (auto& v : data_)
    {
      v *= scalar;
    }
    return *this;
  }

  inline friend Matrix<DataType> operator*(DataType scalar, const Matrix<DataType>& m)
  {
    return m * scalar;
  }

  /** 带列主元选取的高斯-约尔当消元法求逆 */
  Matrix Inverse() const
    requires FloatingPoint<DataType>
  {
    if (rows_ != cols_)
    {
      throw std::invalid_argument("Only square matrices can be inverted");
    }
    const size_t N = rows_;
    Matrix aug(N, 2 * N, DataType(0));
    for (size_t i = 0; i < N; ++i)
    {
      for (size_t j = 0; j < N; ++j)
      {
        aug(i, j) = (*this)(i, j);
      }
      aug(i, N + i) = DataType(1);
    }

    for (size_t col = 0; col < N; ++col)
    {
      size_t pivot = col;
      for (size_t row = col + 1; row < N; ++row)
      {
        if (std::abs(aug(row, col)) > std::abs(aug(pivot, col)))
        {
          pivot = row;
        }
      }

      if (pivot != col)
      {
        for (size_t j = 0; j < 2 * N; ++j)
        {
          std::swap(aug(col, j), aug(pivot, j));
        }
      }

      DataType diag = aug(col, col);
      if (std::abs(diag) < EPSILON)
      {
        throw std::runtime_error("Matrix is singular and cannot be inverted");
      }

      for (size_t j = 0; j < 2 * N; ++j)
      {
        aug(col, j) /= diag;
      }

      for (size_t row = 0; row < N; ++row)
      {
        if (row == col)
        {
          continue;
        }
        DataType factor = aug(row, col);
        for (size_t j = 0; j < 2 * N; ++j)
        {
          aug(row, j) -= factor * aug(col, j);
        }
      }
    }

    Matrix inv(N, N);
    for (size_t i = 0; i < N; ++i)
    {
      for (size_t j = 0; j < N; ++j)
      {
        inv(i, j) = aug(i, N + j);
      }
    }
    return inv;
  }

  /** 带列主元选取的 LU 分解法求行列式 */
  DataType Determinant() const
    requires FloatingPoint<DataType>
  {
    if (rows_ != cols_)
    {
      throw std::invalid_argument("Determinant is only defined for square matrices");
    }

    size_t n = rows_;
    Matrix<DataType> lu(*this);
    DataType det = DataType(1);

    for (size_t col = 0; col < n; ++col)
    {
      size_t pivot = col;
      for (size_t row = col + 1; row < n; ++row)
      {
        if (std::abs(lu(row, col)) > std::abs(lu(pivot, col)))
        {
          pivot = row;
        }
      }

      if (pivot != col)
      {
        for (size_t j = 0; j < n; ++j)
        {
          std::swap(lu(col, j), lu(pivot, j));
        }
        det = -det;
      }

      if (std::abs(lu(col, col)) < EPSILON)
      {
        return DataType(0);
      }

      det *= lu(col, col);

      for (size_t row = col + 1; row < n; ++row)
      {
        DataType factor = lu(row, col) / lu(col, col);
        for (size_t j = col; j < n; ++j)
        {
          lu(row, j) -= factor * lu(col, j);
        }
      }
    }
    return det;
  }

  static Matrix Identity(size_t size)
  {
    Matrix result(size, size, DataType(0));
    for (size_t i = 0; i < size; ++i)
    {
      result(i, i) = DataType(1);
    }
    return result;
  }

  void Print(const std::string& name = "", std::ostream& os = std::cout) const
  {
    if (!name.empty())
    {
      os << name << " =\n";
    }
    for (size_t i = 0; i < rows_; ++i)
    {
      for (size_t j = 0; j < cols_; ++j)
      {
        os << (*this)(i, j) << " ";
      }
      os << "\n";
    }
  }

 private:
};

#endif  // MATRIX_H
