//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_TANH_H
#define CORTESIAN_TANH_H

#include "Activation.h"

class Tanh : public Activation {
public:
  Tanh();
  Eigen::VectorXd function(const Eigen::VectorXd &in) override;
  Eigen::VectorXd derivative(const Eigen::VectorXd &in) override;
};

#endif // CORTESIAN_TANH_H
