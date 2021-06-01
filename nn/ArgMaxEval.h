//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_ARGMAXEVAL_H
#define CORTESIAN_ARGMAXEVAL_H

#include "../libs/Eigen/Core"
#include "EvaluationFunction.h"
class ArgMaxEval : public EvaluationFunction {
public:
  double apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                 const Eigen::VectorXd &Y) override;
  double apply_evaluation(const std::vector<Eigen::VectorXd> &Y_hat,
                          const std::vector<Eigen::VectorXd> &Y) override;
};

#endif // CORTESIAN_ARGMAXEVAL_H
