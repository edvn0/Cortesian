//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_EVALUATIONFUNCTION_H
#define CORTESIAN_EVALUATIONFUNCTION_H

#include <Eigen/Core>
#include <vector>

class EvaluationFunction {
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
                                  const std::vector<Eigen::VectorXd> &Y) = 0;

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
    for (long i = 0; i < size; i++) {
      Eigen::VectorXd col_x = Y_hat[i];
      Eigen::VectorXd col_y = Y_tensor.row(i);
      eval += apply_evaluation_single(col_x, col_y);
    }
    return eval / (double)size;
  }
};

#endif // CORTESIAN_EVALUATIONFUNCTION_H
