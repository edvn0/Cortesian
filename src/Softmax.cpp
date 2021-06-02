//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../include/Softmax.h"

Eigen::VectorXd Softmax::function(Eigen::VectorXd in) { return soft_max(in); }

Eigen::VectorXd Softmax::derivative(Eigen::VectorXd in) { assert(false); }

Eigen::MatrixXd Softmax::derivativeOnInput(Eigen::VectorXd in,
                                           Eigen::VectorXd out) {
  auto sum = (in.array() * out.array()).sum();
  auto diff = out.array() - sum;
  return in.array() * diff;
}

Eigen::VectorXd Softmax::soft_max(const Eigen::VectorXd &in) {
  Eigen::VectorXd max_vector = Eigen::VectorXd::Constant(in.rows(), in.maxCoeff());
  auto z = in - max_vector;
  auto sum = z.unaryExpr([](double t) { return exp(t); }).sum();
  return z.unaryExpr([&](double t) {
    auto val =exp(t) / sum;
    if (abs(val - soft_max_epsilon) < 0) {
      return 0.0;
    } else {
      return val;
    }
  });
}
