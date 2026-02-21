#ifndef TYPES_H
#define TYPES_H

#include <concepts>

template <typename T>
concept MatrixUnit = requires(T a, T b) {
  { a + b } -> std::convertible_to<T>;
  { a - b } -> std::convertible_to<T>;
  { a * b } -> std::convertible_to<T>;
};

template <typename L, typename R>
using CommonType = std::common_type_t<L, R>;

template <typename T>
concept FloatingPoint = std::floating_point<T>;

template <typename L, typename R, typename Result>
concept Addable = requires(L a, R b) {
  { a + b } -> std::convertible_to<Result>;
};

template <typename L, typename R, typename Result>
concept Subtractable = requires(L a, R b) {
  { a - b } -> std::convertible_to<Result>;
};

template <typename L, typename R, typename Result>
concept Multipliable = requires(L a, R b) {
  { a * b } -> std::convertible_to<Result>;
};

template <typename L, typename R, typename Result>
concept Divisible = requires(L a, R b) {
  { a / b } -> std::convertible_to<Result>;
};

#endif  // TYPES_H