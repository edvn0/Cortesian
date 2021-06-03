//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_ARGMAX_H
#define CORTESIAN_ARGMAX_H

#include "../loss_evals/EvaluationFunction.h"

class ArgMax : public EvaluationFunction {
public:
  double apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                 const Eigen::VectorXd &Y) override;
};

#endif // CORTESIAN_ARGMAX_H
