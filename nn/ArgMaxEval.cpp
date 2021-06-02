//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "ArgMaxEval.h"

#include "MathUtils.h"

double ArgMaxEval::apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                           const Eigen::VectorXd &Y) {
  return arg_max(Y_hat) == arg_max(Y) ? 1.0 : 0.0;
}

double ArgMaxEval::apply_evaluation(const std::vector<Eigen::VectorXd> &Y_hat,
                                    const std::vector<Eigen::VectorXd> &Y) {
  size_t correct = 0;
  size_t len = Y_hat.size();

  for (size_t i = 0; i < len; i++) {
    correct += apply_evaluation_single(Y_hat[i], Y[i]) > 0 ? 1 : 0;
  }

  return (double)correct / (double)len;
}
