//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_EVALUATIONFUNCTION_H
#define CORTESIAN_EVALUATIONFUNCTION_H

#include "../libs/Eigen/Core"
#include <vector>

class EvaluationFunction {
public:
  virtual ~EvaluationFunction() = default;
  virtual double apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                         const Eigen::VectorXd &Y) = 0;
  virtual double apply_evaluation(const std::vector<Eigen::VectorXd> &Y_hat,
                                  const std::vector<Eigen::VectorXd> &Y) = 0;
};

#endif // CORTESIAN_EVALUATIONFUNCTION_H
