//
// Created by Edwin Carlsson on 2021-06-07.
//

#ifndef CORTESIAN_LOGLOSS_H
#define CORTESIAN_LOGLOSS_H

#include "../../include/loss_evals/LossFunction.h"

class LogLoss : public LossFunction {
public:
  double apply_loss_single(const Eigen::VectorXd &Y_hat,
                           const Eigen::VectorXd &y) override;

  Eigen::MatrixXd apply_loss_gradient(const Eigen::MatrixXd &y_hat,
                                      const Eigen::MatrixXd &y) override;
};

#endif // CORTESIAN_LOGLOSS_H
