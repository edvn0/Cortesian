//
// Created by Edwin Carlsson on 2021-06-02.
//

#include "../../include/loss_evals/CategoricalCrossEntropy.h"

#include <iostream>

double
CategoricalCrossEntropy::apply_loss(const std::vector<Eigen::VectorXd> &Y_hat,
                                    const std::vector<Eigen::VectorXd> &Y) {
  return -1.0 * LossFunction::calculate(Y_hat, Y);
}

double CategoricalCrossEntropy::apply_loss_single(const Eigen::VectorXd &Y_hat,
                                                  const Eigen::VectorXd &y) {
  auto inner = (y.array() * (Y_hat.array() + 1e-9).log());
  auto sum = inner.sum();
  return sum;
}

Eigen::MatrixXd
CategoricalCrossEntropy::apply_loss_gradient(const Eigen::MatrixXd &y_hat,
                                             const Eigen::MatrixXd &y) {
  return y_hat - y;
}

double
CategoricalCrossEntropy::apply_loss(const std::vector<Eigen::VectorXd> &Y_hat,
                                    const Eigen::MatrixXd &Y) {
  return -1.0 * LossFunction::calculate(Y_hat, Y);
}
