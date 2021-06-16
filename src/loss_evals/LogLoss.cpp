//
// Created by Edwin Carlsson on 2021-06-07.
//

#include "../../include/loss_evals/LogLoss.h"

double LogLoss::apply_loss_single(const Eigen::VectorXd &Y_hat,
                                  const Eigen::VectorXd &y) {
  // these two have size 2
  return -1.0 * (y.array() * (Y_hat.array() + 1e-9).log()).sum();
}
Eigen::MatrixXd LogLoss::apply_loss_gradient(const Eigen::MatrixXd &y_hat,
                                             const Eigen::MatrixXd &y) {
  return y_hat - y;
}
