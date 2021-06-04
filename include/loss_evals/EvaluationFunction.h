//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_EVALUATIONFUNCTION_H
#define CORTESIAN_EVALUATIONFUNCTION_H

#include "../utils/MetaBase.h"
#include <eigen3/Eigen/Core>
#include <vector>

class EvaluationFunction : public MetaBase {
public:
  virtual ~EvaluationFunction() = default;

  virtual double apply_evaluation_single(const Eigen::VectorXd &Y_hat,
                                         const Eigen::VectorXd &Y) = 0;

  /**
   * Applies evaluation metrics against a vector of Eigen::Vectors.
   * @param Y_hat predicted Xs -> Y_hat.
   * @param Y real Y
   * @return a metric score
   */
  virtual double apply_evaluation(const std::vector<Eigen::VectorXd> &Y_hat,
                                  const std::vector<Eigen::VectorXd> &Y) {
    double eval = 0.0;
    size_t size = Y_hat.size();
#pragma omp parallel for
    for (long i = 0; i < size; i++) {
      eval += apply_evaluation_single(Y_hat[i], Y[i]);
    }
    return eval / (double)size;
  };

  /**
   * Overload of {@code #apply_evaluation} instead taking Y as a tensor.
   * @param Y_hat predicted Xs -> Y_hat
   * @param Y_tensor real Ys, as tensor
   * @return a metric score
   */
  virtual double apply_evaluation(const std::vector<Eigen::VectorXd> &Y_hat,
                                  const Eigen::MatrixXd &Y_tensor) {
    double eval = 0.0;
    size_t size = Y_hat.size();
#pragma omp parallel for
    for (long i = 0; i < size; i++) {
      eval += apply_evaluation_single(Y_hat[i], Y_tensor.row(i));
    }
    return eval / (double)size;
  }
};

#endif // CORTESIAN_EVALUATIONFUNCTION_H
