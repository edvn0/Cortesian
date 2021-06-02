//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../../include/loss_evals/MeanSquared.h"

double MeanSquared::apply_loss(const std::vector<Eigen::VectorXd> &X,
                               const std::vector<Eigen::VectorXd> &Y) {
  double loss = 0.0;
  size_t len = X.size();
  for (size_t i = 0; i < len; ++i) {
    loss += apply_loss_single(X[i], Y[i]);
  }
  return loss / (double)len;
}

double MeanSquared::apply_loss_single(const Eigen::VectorXd &x,
                                      const Eigen::VectorXd &y) {
  auto diff = (x - y).array();
  return (diff * diff).sum();
}

Eigen::MatrixXd MeanSquared::apply_loss_gradient(const Eigen::MatrixXd &y_hat,
                                                 const Eigen::MatrixXd &y) {
  return y_hat - y;
}

double MeanSquared::apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                            const Eigen::VectorXd &Y) {
  return apply_loss_single(Y_hat, Y);
}

double MeanSquared::apply_evaluation(const std::vector<Eigen::VectorXd> &Y_hat,
                                     const std::vector<Eigen::VectorXd> &Y) {
  return apply_loss(Y_hat, Y);
}
