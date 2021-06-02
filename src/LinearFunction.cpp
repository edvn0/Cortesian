//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../include/LinearFunction.h"
Eigen::VectorXd LinearFunction::function(Eigen::VectorXd in) { return in; }
Eigen::VectorXd LinearFunction::derivative(Eigen::VectorXd in) {
  return Eigen::VectorXd::Ones(in.rows());
}
