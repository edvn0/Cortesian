//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_TANH_H
#define CORTESIAN_TANH_H

#include "Activation.h"

class Tanh : public Activation {
public:
  Eigen::VectorXd function(Eigen::VectorXd in) override;
  Eigen::VectorXd derivative(Eigen::VectorXd in) override;
};

#endif // CORTESIAN_TANH_H
