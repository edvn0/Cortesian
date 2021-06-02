//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_LINEARFUNCTION_H
#define CORTESIAN_LINEARFUNCTION_H

#include "Activation.h"

class LinearFunction : public Activation {
 public:
  Eigen::VectorXd function(Eigen::VectorXd in) override;
  Eigen::VectorXd derivative(Eigen::VectorXd in) override;
};

#endif  // CORTESIAN_LINEARFUNCTION_H
