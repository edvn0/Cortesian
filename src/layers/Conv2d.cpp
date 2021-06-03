//
// Created by Edwin Carlsson on 2021-06-03.
//

#include "../../include/layers/Conv2d.h"

Eigen::VectorXd Conv2D::calculate(const Eigen::VectorXd &in) {
  return Eigen::VectorXd();
}

void Conv2D::add_deltas(const Eigen::MatrixXd &d_w,
                        const Eigen::VectorXd &d_b) {}

void Conv2D::fit(Optimizer *optimizer) {}

Eigen::VectorXd Conv2D::activate(Eigen::VectorXd in, Eigen::VectorXd out) {
  return Eigen::VectorXd();
}

Eigen::MatrixXd Conv2D::error_derivative(const Eigen::VectorXd &previous,
                                         const Eigen::MatrixXd &current) {
  return Eigen::MatrixXd();
}

Eigen::VectorXd Conv2D::calculate(Eigen::VectorXd &in) {
  return Eigen::VectorXd();
}

Eigen::MatrixXd Conv2D::get_weight() { return Eigen::MatrixXd(); }

Eigen::MatrixXd Conv2D::get_activated() { return Eigen::MatrixXd(); }

void Conv2D::set_params(Eigen::MatrixXd weight, Eigen::VectorXd bias) {}

void Conv2D::set_delta_params(Eigen::MatrixXd d_weight,
                              Eigen::VectorXd d_bias) {}
