#include "Eigen/Dense"
#include "KF.hpp"

int main()
{
  Matrix<double> a(4, 3, 2.5);
  Matrix<int> b(3, 4, 5);

  auto c = b * a;
  std::cout << "Result of a * b:\n";
  c.Print();

  Eigen::Matrix<double, 4, 3> e(4, 3);
  Eigen::Matrix<double, 3, 4> f(3, 4);
  e.setConstant(2.5);
  f.setConstant(5);
  auto g = f * e;
  std::cout << "Eigen result:\n" << g << '\n';
  return 0;
}