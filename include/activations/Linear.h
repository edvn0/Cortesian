//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_LINEAR_H
#define CORTESIAN_LINEAR_H

#include "Activation.h"

class Linear : public Activation {
public:
  Linear();
  Eigen::VectorXd function(const Eigen::VectorXd& in) override;
  Eigen::VectorXd derivative(const Eigen::VectorXd& in) override;
};

#endif // CORTESIAN_LINEAR_H
