//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_SIGMOIDFUNCTION_H
#define CORTESIAN_SIGMOIDFUNCTION_H

#include "Activation.h"

class SigmoidFunction: public Activation {
private:
  static double approx(double t);
public:
  Eigen::VectorXd function(Eigen::VectorXd in) override;
  Eigen::VectorXd derivative(Eigen::VectorXd in) override;
};

#endif // CORTESIAN_SIGMOIDFUNCTION_H
