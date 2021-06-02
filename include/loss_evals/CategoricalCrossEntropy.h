//
// Created by Edwin Carlsson on 2021-06-02.
//

#ifndef CORTESIAN_CATEGORICALCROSSENTROPY_H
#define CORTESIAN_CATEGORICALCROSSENTROPY_H

#include "LossFunction.h"

class CategoricalCrossEntropy : public LossFunction {
public:
  double apply_loss(const std::vector<Eigen::VectorXd> &Y_hat,
                    const std::vector<Eigen::VectorXd> &Y) override;

  double apply_loss_single(const Eigen::VectorXd &Y_hat,
                           const Eigen::VectorXd &y) override;

  Eigen::MatrixXd apply_loss_gradient(const Eigen::MatrixXd &y_hat,
                                      const Eigen::MatrixXd &y) override;
  double apply_loss(const std::vector<Eigen::VectorXd> &Y_hat,
                    const Eigen::MatrixXd &Y) override;
};

#endif // CORTESIAN_CATEGORICALCROSSENTROPY_H
