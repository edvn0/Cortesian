//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_LOSSFUNCTION_H
#define CORTESIAN_LOSSFUNCTION_H

class LossFunction {
public:
  virtual ~LossFunction() = default;

  /**
   * Applies the loss function to an already predicted data set.
   * Caller must have predicted the data to call this function.
   * @param Y_hat predicted data set
   * @param Y real values
   * @return the loss
   */
  virtual double apply_loss(const std::vector<Eigen::VectorXd>& Y_hat, const std::vector<Eigen::VectorXd>& Y) = 0;

  /**
   * Applies the loss function to an already predicted data set.
   * Caller must have predicted the data to call this function.
   * @param Y_hat predicted
   * @param y real
   * @return loss
   */
  virtual double apply_loss_single(const Eigen::VectorXd& Y_hat, const Eigen::VectorXd& y) = 0;

  /**
   * Calculates the gradient vector from the predicted/real pair.
   * @param y_hat predicted X
   * @param y real(X)
   * @return gradient vector.
   */
  virtual Eigen::MatrixXd apply_loss_gradient(const Eigen::MatrixXd& y_hat, const Eigen::MatrixXd& y) = 0;
};

#endif // CORTESIAN_LOSSFUNCTION_H