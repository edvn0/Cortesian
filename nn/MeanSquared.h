//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_MEANSQUARED_H
#define CORTESIAN_MEANSQUARED_H

#include "EvaluationFunction.h"
#include "LossFunction.h"

class MeanSquared : public LossFunction, public EvaluationFunction {
 public:
  double apply_loss(const std::vector<Eigen::VectorXd> &X,
                    const std::vector<Eigen::VectorXd> &Y) override;
  double apply_loss_single(const Eigen::VectorXd &x,
                           const Eigen::VectorXd &y) override;
  Eigen::MatrixXd apply_loss_gradient(const Eigen::MatrixXd &y_hat,
                                      const Eigen::MatrixXd &y) override;
  double apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                 const Eigen::VectorXd &Y) override;
  double apply_evaluation(const std::vector<Eigen::VectorXd> &Y_hat,
                          const std::vector<Eigen::VectorXd> &Y) override;
};

#endif  // CORTESIAN_MEANSQUARED_H
