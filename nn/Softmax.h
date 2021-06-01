//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_SOFTMAX_H
#define CORTESIAN_SOFTMAX_H

#include "Activation.h"

class Softmax: public Activation {
private:
  Eigen::VectorXd soft_max(const Eigen::VectorXd& in);
public:
  Eigen::VectorXd function(Eigen::VectorXd in) override;
  Eigen::VectorXd derivative(Eigen::VectorXd in) override;
  Eigen::MatrixXd derivativeOnInput(Eigen::VectorXd in,
                                    Eigen::VectorXd out) override;
};

#endif // CORTESIAN_SOFTMAX_H
