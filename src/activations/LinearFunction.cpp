//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../../include/activations/LinearFunction.h"
Eigen::VectorXd LinearFunction::function(Eigen::VectorXd in) { return in; }

Eigen::VectorXd LinearFunction::derivative(Eigen::VectorXd in) {
  return Eigen::VectorXd::Ones(in.rows());
}

LinearFunction::LinearFunction() {
  this->operator()("activation", "LinearFunction");
}
