//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../../include/activations/Linear.h"
Eigen::VectorXd Linear::function(const Eigen::VectorXd &in) { return in; }

Eigen::VectorXd Linear::derivative(const Eigen::VectorXd &in) {
  return Eigen::VectorXd::Ones(in.rows());
}

Linear::Linear() { this->operator()("activation", "Linear"); }
