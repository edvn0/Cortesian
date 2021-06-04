//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../../include/loss_evals/ArgMax.h"

#include "../../include/utils/MathUtils.h"

double ArgMax::apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                       const Eigen::VectorXd &Y) {
  return arg_max(Y_hat) == arg_max(Y) ? 1.0 : 0.0;
}
