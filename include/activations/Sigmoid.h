//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_SIGMOID_H
#define CORTESIAN_SIGMOID_H

#include "Activation.h"

class Sigmoid : public Activation {
private:
  static double approx(double t);

public:
  Sigmoid();
  Eigen::VectorXd function(const Eigen::VectorXd& in) override;
  Eigen::VectorXd derivative(const Eigen::VectorXd& in) override;
};

#endif // CORTESIAN_SIGMOID_H
