#ifndef MATRIX_H
#define MATRIX_H

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "Types.hpp"

template <MatrixUnit DataType = double>
class Matrix
{
 private:
  size_t rows_;
  size_t cols_;
  std::vector<DataType> data_;

  constexpr static DataType DEFAULT_EPSILON = []()
  {
    if constexpr (std::floating_point<DataType>)
    {
      return std::numeric_limits<DataType>::epsilon() * DataType(100);
    }
    else
    {
      return DataType(0);
    }
  }();

  /** 以当前列最大绝对值为参考的相对阈值 */
  static DataType RelativeEpsilon(DataType col_max_abs)
    requires FloatingPoint<DataType>
  {
    return DEFAULT_EPSILON * std::max(col_max_abs, std::numeric_limits<DataType>::min());
  }

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

  Matrix(size_t rows, size_t cols, std::initializer_list<DataType> init)
      : rows_(rows), cols_(cols), data_(rows * cols, DataType(0))
  {
    if (rows == 0 || cols == 0)
    {
      throw std::invalid_argument("Matrix dimensions must be positive");
    }
    if (init.size() > rows * cols)
    {
      throw std::invalid_argument("Too many initializer elements");
    }
    std::copy(init.begin(), init.end(), data_.begin());
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
      throw std::out_of_range("Matrix index out of range: (" + std::to_string(row) +
                              ", " + std::to_string(col) + ")");
    }
    return data_[row * cols_ + col];
  }

  const DataType& At(size_t row, size_t col) const
  {
    if (row >= rows_ || col >= cols_)
    {
      throw std::out_of_range("Matrix index out of range: (" + std::to_string(row) +
                              ", " + std::to_string(col) + ")");
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

  bool operator==(const Matrix<DataType>& other) const
  {
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
      return false;
    }
    for (size_t i = 0; i < data_.size(); ++i)
    {
      if (data_[i] != other.data_[i])
      {
        return false;
      }
    }
    return true;
  }

  bool operator!=(const Matrix<DataType>& other) const { return !(*this == other); }

  bool ApproxEqual(const Matrix<DataType>& other, double tol = 1e-9) const
    requires FloatingPoint<DataType>
  {
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
      return false;
    }
    for (size_t i = 0; i < data_.size(); ++i)
    {
      if (std::abs(static_cast<double>(data_[i]) - static_cast<double>(other.data_[i])) >
          tol)
      {
        return false;
      }
    }
    return true;
  }

  template <MatrixUnit OtherType>
    requires Addable<DataType, OtherType, CommonType<DataType, OtherType>>
  Matrix<CommonType<DataType, OtherType>> operator+(const Matrix<OtherType>& other) const
  {
    using ResultType = CommonType<DataType, OtherType>;
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
      throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    Matrix<ResultType> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i)
    {
      for (size_t j = 0; j < cols_; ++j)
      {
        result(i, j) =
            static_cast<ResultType>((*this)(i, j)) + static_cast<ResultType>(other(i, j));
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
    requires Addable<DataType, OtherType, CommonType<DataType, OtherType>>
  friend Matrix<CommonType<DataType, OtherType>> operator+(Matrix<DataType>&& lhs,
                                                           const Matrix<OtherType>& rhs)
  {
    using ResultType = CommonType<DataType, OtherType>;
    if constexpr (std::is_same_v<DataType, ResultType>)
    {
      lhs += rhs;
      return std::move(lhs);
    }
    else
    {
      return static_cast<const Matrix<DataType>&>(lhs) + rhs;
    }
  }

  template <MatrixUnit OtherType>
    requires Subtractable<DataType, OtherType, CommonType<DataType, OtherType>>
  Matrix<CommonType<DataType, OtherType>> operator-(const Matrix<OtherType>& other) const
  {
    using ResultType = CommonType<DataType, OtherType>;
    if (rows_ != other.rows_ || cols_ != other.cols_)
    {
      throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    Matrix<ResultType> result(rows_, cols_);
    for (size_t i = 0; i < rows_; ++i)
    {
      for (size_t j = 0; j < cols_; ++j)
      {
        result(i, j) =
            static_cast<ResultType>((*this)(i, j)) - static_cast<ResultType>(other(i, j));
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
    requires Subtractable<DataType, OtherType, CommonType<DataType, OtherType>>
  friend Matrix<CommonType<DataType, OtherType>> operator-(Matrix<DataType>&& lhs,
                                                           const Matrix<OtherType>& rhs)
  {
    using ResultType = CommonType<DataType, OtherType>;
    if constexpr (std::is_same_v<DataType, ResultType>)
    {
      lhs -= rhs;
      return std::move(lhs);
    }
    else
    {
      return static_cast<const Matrix<DataType>&>(lhs) - rhs;
    }
  }

  template <MatrixUnit OtherType>
    requires Multipliable<DataType, OtherType, CommonType<DataType, OtherType>>
  Matrix<CommonType<DataType, OtherType>> operator*(const Matrix<OtherType>& other) const
  {
    using ResultType = CommonType<DataType, OtherType>;
    if (cols_ != other.rows_)
    {
      throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }

    Matrix<ResultType> result(rows_, other.cols_, ResultType(0));
    for (size_t i = 0; i < rows_; ++i)
    {
      for (size_t j = 0; j < cols_; ++j)
      {
        auto a_ij = static_cast<ResultType>((*this)(i, j));
        for (size_t k = 0; k < other.cols_; ++k)
        {
          result(i, k) += a_ij * static_cast<ResultType>(other(j, k));
        }
      }
    }
    return result;
  }

  template <typename Scalar>
    requires Multipliable<DataType, Scalar, CommonType<DataType, Scalar>>
  Matrix<CommonType<DataType, Scalar>> operator*(Scalar scalar) const
  {
    using R = CommonType<DataType, Scalar>;
    Matrix<R> result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i)
    {
      result.data_[i] = static_cast<R>(data_[i]) * static_cast<R>(scalar);
    }
    return result;
  }

  template <typename Scalar>
    requires Multipliable<Scalar, DataType, CommonType<Scalar, DataType>>
  inline friend Matrix<CommonType<Scalar, DataType>> operator*(Scalar scalar,
                                                               const Matrix<DataType>& m)
  {
    return m * scalar;
  }

  Matrix<DataType>& operator*=(DataType scalar)
  {
    for (auto& v : data_)
    {
      v *= scalar;
    }
    return *this;
  }

  template <typename Scalar>
    requires Divisible<DataType, Scalar, CommonType<DataType, Scalar>>
  Matrix<CommonType<DataType, Scalar>> operator/(Scalar scalar) const
  {
    using ResultType = CommonType<DataType, Scalar>;
    if constexpr (std::floating_point<Scalar>)
    {
      if (std::abs(scalar) < std::numeric_limits<Scalar>::min())
      {
        throw std::invalid_argument("Division by near-zero scalar");
      }
    }
    else
    {
      if (scalar == Scalar(0))
      {
        throw std::invalid_argument("Division by zero");
      }
    }
    Matrix<ResultType> result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i)
    {
      result.data_[i] =
          static_cast<ResultType>(data_[i]) / static_cast<ResultType>(scalar);
    }
    return result;
  }

  Matrix<DataType>& operator/=(DataType scalar)
  {
    if constexpr (std::floating_point<DataType>)
    {
      if (std::abs(scalar) < std::numeric_limits<DataType>::min())
      {
        throw std::invalid_argument("Division by near-zero scalar");
      }
    }
    else
    {
      if (scalar == DataType(0))
      {
        throw std::invalid_argument("Division by zero");
      }
    }
    for (auto& v : data_)
    {
      v /= scalar;
    }
    return *this;
  }

  Matrix<DataType> operator-() const
  {
    Matrix<DataType> result(rows_, cols_);
    for (size_t i = 0; i < data_.size(); ++i)
    {
      result.data_[i] = -data_[i];
    }
    return result;
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
      DataType max_abs = std::abs(aug(col, col));
      for (size_t row = col + 1; row < N; ++row)
      {
        DataType abs_val = std::abs(aug(row, col));
        if (abs_val > max_abs)
        {
          max_abs = abs_val;
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

      if (std::abs(diag) < RelativeEpsilon(max_abs))
      {
        throw std::runtime_error(
            "Matrix is singular or nearly singular and cannot be inverted");
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
      DataType max_abs = std::abs(lu(col, col));
      for (size_t row = col + 1; row < n; ++row)
      {
        DataType abs_val = std::abs(lu(row, col));
        if (abs_val > max_abs)
        {
          max_abs = abs_val;
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

      if (std::abs(lu(col, col)) < RelativeEpsilon(max_abs))
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
};

#endif  // MATRIX_H
