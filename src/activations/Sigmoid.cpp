//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../../include/activations/Sigmoid.h"

double Sigmoid::approx(double t) {
  if (t < -2.0) {
    return -1.0;
  } else if (t > 2.0) {
    return 1.0;
  } else {
    return exp(t) / (exp(t) + 1);
  }
}

Eigen::VectorXd Sigmoid::function(const Eigen::VectorXd &in) {
  return in.unaryExpr([&](double t) { return approx(t); });
}

Eigen::VectorXd Sigmoid::derivative(const Eigen::VectorXd &in) {
  return in.array() * (1 - in.array());
}

Sigmoid::Sigmoid() { this->operator()("activation", "Sigmoid"); }
