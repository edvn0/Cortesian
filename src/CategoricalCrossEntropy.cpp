//
// Created by Edwin Carlsson on 2021-06-02.
//

#include "../include/CategoricalCrossEntropy.h"

#include <iostream>

double
CategoricalCrossEntropy::apply_loss(const std::vector<Eigen::VectorXd> &Y_hat,
                                    const std::vector<Eigen::VectorXd> &Y) {
  double loss = 0.0;
  size_t rows = Y_hat.size();
  for (size_t i = 0; i < rows; i++) {
    loss += apply_loss_single(Y_hat[i], Y[i]);
  }
  return (-1.0 * loss) / (double) rows;
}

double CategoricalCrossEntropy::apply_loss_single(const Eigen::VectorXd &Y_hat,
                                                  const Eigen::VectorXd &y) {

  auto logged = Y_hat.array().log();
  auto mult = y.array() * logged;
  return mult.sum();
}

Eigen::MatrixXd
CategoricalCrossEntropy::apply_loss_gradient(const Eigen::MatrixXd &y_hat,
                                             const Eigen::MatrixXd &y) {
  return y_hat - y;
}

double
CategoricalCrossEntropy::apply_loss(const std::vector<Eigen::VectorXd> &Y_hat,
                                    const Eigen::MatrixXd &Y) {
  double loss = 0.0;
  size_t rows = Y_hat.size();
  for (size_t i = 0; i < rows; i++) {
    loss += apply_loss_single(Y_hat[i], Y.row((long)i));
  }
  return (-1.0 * loss) / (double) rows;
}
