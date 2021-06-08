//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../../include/loss_evals/ArgMax.h"

#include "../../include/utils/MathUtils.h"

double ArgMax::apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                       const Eigen::VectorXd &Y) {

  auto y_hat_am = arg_max(Y_hat);
  auto y_am = arg_max(Y);
  return y_hat_am == y_am ? 100.0 : 0.0;
}
