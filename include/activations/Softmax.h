//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_SOFTMAX_H
#define CORTESIAN_SOFTMAX_H

#include "Activation.h"

class Softmax : public Activation {
private:
  static Eigen::VectorXd soft_max(const Eigen::VectorXd &in);
  static double constexpr soft_max_epsilon = 1e-8;

public:
  Softmax();
  Eigen::VectorXd function(const Eigen::VectorXd &in) override;
  Eigen::VectorXd derivative(const Eigen::VectorXd &in) override;
  Eigen::MatrixXd derivative_on_input(const Eigen::VectorXd &,
                                      const Eigen::VectorXd &) override;
};

#endif // CORTESIAN_SOFTMAX_H
