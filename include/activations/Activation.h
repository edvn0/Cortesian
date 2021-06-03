//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_ACTIVATION_H
#define CORTESIAN_ACTIVATION_H

#include <eigen3/Eigen/Core>

class Activation {
public:
  virtual ~Activation() = default;
  virtual Eigen::VectorXd function(Eigen::VectorXd in) = 0;
  virtual Eigen::VectorXd derivative(Eigen::VectorXd in) = 0;

  virtual Eigen::MatrixXd derivative_on_input(Eigen::VectorXd in,
                                            Eigen::VectorXd out) {
    auto arrPred = out.array();
    auto arrRaw = in.array();
    Eigen::MatrixXd fixed = arrRaw * arrPred;
    auto diffRaw = derivative(arrRaw);
    return arrPred * diffRaw.array();
  };
};

#endif // CORTESIAN_ACTIVATION_H
