//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_ARGMAX_H
#define CORTESIAN_ARGMAX_H

#include "EvaluationFunction.h"

class ArgMax : public EvaluationFunction {
public:
  ArgMax() { this->operator()("evaluation", "ArgMax"); }
  double apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                 const Eigen::VectorXd &Y) override;
};

#endif // CORTESIAN_ARGMAX_H
