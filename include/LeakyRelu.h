//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_LEAKYRELU_H
#define CORTESIAN_LEAKYRELU_H

#include "Activation.h"

class LeakyRelu : public Activation {
  double cap{0.01};

 public:
  Eigen::VectorXd function(Eigen::VectorXd in) override;

  Eigen::VectorXd derivative(Eigen::VectorXd in) override;
};

#endif  // CORTESIAN_LEAKYRELU_H
