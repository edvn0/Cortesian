//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_LOSSFUNCTION_H
#define CORTESIAN_LOSSFUNCTION_H

#include "../utils/MetaBase.h"
#include <eigen3/Eigen/Core>
#include <vector>

class LossFunction : public MetaBase {
public:
  virtual ~LossFunction() = default;
protected:
  virtual double calculate(const std::vector<Eigen::VectorXd> &Y_hat,
                           const std::vector<Eigen::VectorXd> &Y) {
    double loss = 0.0;
    size_t rows = Y_hat.size();
    for (size_t i = 0; i < rows; i++) {
      loss += apply_loss_single(Y_hat[i], Y[i]);
    }
    return loss / (double)rows;
  }

  virtual double calculate(const std::vector<Eigen::VectorXd> &Y_hat,
                           const Eigen::MatrixXd &Y_tensor) {
    double loss = 0.0;
    size_t rows = Y_hat.size();
    for (size_t i = 0; i < rows; i++) {
      loss += apply_loss_single(Y_hat[i], Y_tensor.row((long)i));
    }
    return loss / (double)rows;
  }

public:

  /**
   * Applies the loss function to an already predicted data set.
   * Caller must have predicted the data to call this function.
   * @param Y_hat predicted data set
   * @param Y real values
   * @return the loss
   */
  virtual double apply_loss(const std::vector<Eigen::VectorXd> &Y_hat,
                            const std::vector<Eigen::VectorXd> &Y) {
    return calculate(Y_hat, Y);
  };

  /**
   * Applies the loss function to an already tensor
   * Caller must have predicted the data to call this function.
   * @param Y_hat predicted data set
   * @param Y real values
   * @return the loss
   */
  virtual double apply_loss(const std::vector<Eigen::VectorXd> &Y_hat,
                            const Eigen::MatrixXd &Y) {
    return calculate(Y_hat, Y);
  }

  /**
   * Applies the loss function to an already predicted data set.
   * Caller must have predicted the data to call this function.
   * @param Y_hat predicted
   * @param y real
   * @return loss
   */
  virtual double apply_loss_single(const Eigen::VectorXd &Y_hat,
                                   const Eigen::VectorXd &y) = 0;

  /**
   * Calculates the gradient vector from the predicted/real pair.
   * @param y_hat predicted X
   * @param y real(X)
   * @return gradient vector.
   */
  virtual Eigen::MatrixXd apply_loss_gradient(const Eigen::MatrixXd &y_hat,
                                              const Eigen::MatrixXd &y) = 0;
};

#endif // CORTESIAN_LOSSFUNCTION_H
