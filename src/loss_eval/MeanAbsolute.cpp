//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../../include/loss_evals/MeanAbsolute.h"
double MeanAbsolute::apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                             const Eigen::VectorXd &Y) {
  return apply_loss_single(Y_hat, Y);
}
double MeanAbsolute::apply_evaluation(const std::vector<Eigen::VectorXd> &Y_hat,
                                      const std::vector<Eigen::VectorXd> &Y) {
  return apply_loss(Y_hat, Y);
}
double MeanAbsolute::apply_loss(const std::vector<Eigen::VectorXd> &Y_hat,
                                const std::vector<Eigen::VectorXd> &Y) {
  double loss = 0.0;
  size_t len = Y_hat.size();
  for (size_t i = 0; i < len; ++i) {
    loss += apply_loss_single(Y_hat[i], Y[i]);
  }
  return loss / (double)len;
}

double MeanAbsolute::apply_loss_single(const Eigen::VectorXd &Y_hat,
                                       const Eigen::VectorXd &y) {
  auto diff = Y_hat.array() - y.array();
  return diff.abs().sum();
}

Eigen::MatrixXd MeanAbsolute::apply_loss_gradient(const Eigen::MatrixXd &y_hat,
                                                  const Eigen::MatrixXd &y) {
  auto diff = (y_hat.array() - y.array()).abs();
  Eigen::MatrixXd absed = diff.unaryExpr([&](double t) {
    if (t - 0 < 0.000000001) {
      return 0.0;
    } else if (t > 1) {
      return 1.0;
    } else {
      return -1.0;
    }
  });
  return absed;
}
