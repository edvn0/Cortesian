//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_MEANABSOLUTE_H
#define CORTESIAN_MEANABSOLUTE_H

#include "EvaluationFunction.h"
#include "LossFunction.h"

class MeanAbsolute: public LossFunction, public EvaluationFunction {
public:
  double apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                 const Eigen::VectorXd &Y) override;
  double apply_evaluation(const std::vector<Eigen::VectorXd> &Y_hat,
                          const std::vector<Eigen::VectorXd> &Y) override;
  double apply_loss(const std::vector<Eigen::VectorXd> &Y_hat,
                    const std::vector<Eigen::VectorXd> &Y) override;
  double apply_loss_single(const Eigen::VectorXd &Y_hat,
                           const Eigen::VectorXd &y) override;
  Eigen::MatrixXd apply_loss_gradient(const Eigen::MatrixXd &y_hat,
                                      const Eigen::MatrixXd &y) override;
};

#endif // CORTESIAN_MEANABSOLUTE_H
