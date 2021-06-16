//
// Created by Edwin Carlsson on 2021-05-31.
//

#include "../../include/activations/LeakyRelu.h"

Eigen::VectorXd LeakyRelu::function(const Eigen::VectorXd &in) {
  double &val = cap;
  static const auto func = [&val](double f) {
    if (f > 0) {
      return f;
    } else {
      return val;
    }
  };
  return in.unaryExpr(func);
}

Eigen::VectorXd LeakyRelu::derivative(const Eigen::VectorXd &in) {
  double &val = cap;
  static const auto func = [&val](double f) {
    if (f > 0) {
      return 1.0;
    } else {
      return val;
    }
  };
  return in.unaryExpr(func);
}

LeakyRelu::LeakyRelu() {
  this->operator()("activation", "LeakyRelu");
  this->operator()("alpha", std::to_string(cap));
}
