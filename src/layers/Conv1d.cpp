//
// Created by Edwin Carlsson on 2021-06-03.
//

#include "../../include/layers/Conv1d.h"

Eigen::VectorXd Conv1D::calculate(const Eigen::VectorXd &in) {
  for (int i = -1; i < in.rows() + 1; i++) {
    auto convolution = 0.0;
  }

  return Eigen::VectorXd();
}
void Conv1D::add_deltas(const Eigen::MatrixXd &d_w,
                        const Eigen::VectorXd &d_b) {}
void Conv1D::fit(Optimizer *optimizer) {}
Eigen::VectorXd Conv1D::activate(Eigen::VectorXd in, Eigen::VectorXd out) {
  return Eigen::VectorXd();
}
Eigen::MatrixXd Conv1D::error_derivative(const Eigen::VectorXd &previous,
                                         const Eigen::MatrixXd &current) {
  return Eigen::MatrixXd();
}
Eigen::VectorXd Conv1D::calculate(Eigen::VectorXd &in) {
  return Eigen::VectorXd();
}
Eigen::MatrixXd Conv1D::previous_activation() { return Eigen::MatrixXd(); }
Eigen::MatrixXd Conv1D::get_weight() { return Eigen::MatrixXd(); }
Eigen::MatrixXd Conv1D::get_activated() { return Eigen::MatrixXd(); }
void Conv1D::set_params(Eigen::MatrixXd weight, Eigen::VectorXd bias) {}
void Conv1D::set_delta_params(Eigen::MatrixXd d_weight,
                              Eigen::VectorXd d_bias) {}
