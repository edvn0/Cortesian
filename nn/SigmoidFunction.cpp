//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "SigmoidFunction.h"
double SigmoidFunction::approx(double t) {
  if (t < -2.0) {
    return -1.0;
  } else if (t > 2.0) {
    return 1.0;
  } else {
    return exp(t) / (exp(t) + 1);
  }
}

Eigen::VectorXd SigmoidFunction::function(Eigen::VectorXd in) {
  return in.unaryExpr([&](double t) { return approx(t); });
}

Eigen::VectorXd SigmoidFunction::derivative(Eigen::VectorXd in) {
  return in.array() * (1 - in.array());
}
