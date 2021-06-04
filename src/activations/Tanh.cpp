//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../../include/activations/Tanh.h"

Eigen::VectorXd Tanh::function(Eigen::VectorXd in) {
  return in.unaryExpr([](double t) {
    if (t < -1.0) {
      return -1.0;
    } else if (t > 1.0) {
      return 1.0;
    } else {
      return tanh(t);
    }
  });
}

Eigen::VectorXd Tanh::derivative(Eigen::VectorXd in) {
  return 1 - in.array() * in.array();
}

Tanh::Tanh() { this->operator()("activation", "Tanh"); }
