//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../../include/activations/Softmax.h"
#include <iostream>

Eigen::VectorXd Softmax::function(const Eigen::VectorXd &in) {
  return soft_max(in);
}

Eigen::VectorXd Softmax::derivative(const Eigen::VectorXd &in) {
  assert(false);
}

Eigen::MatrixXd Softmax::derivative_on_input(const Eigen::VectorXd &in,
                                             const Eigen::VectorXd &out) {
  auto sum = (in.array() * out.array()).sum();
  auto diff = out.array() - sum;
  return in.array() * diff;
}

Eigen::VectorXd Softmax::soft_max(const Eigen::VectorXd &in) {
  auto d = Eigen::VectorXd::Constant(in.rows(), in.maxCoeff());
  auto e_x = (in.array() - d.array()).exp();
  auto sum = e_x.sum();
  return e_x / sum;
}

Softmax::Softmax() { this->operator()("activation", "Softmax"); }
