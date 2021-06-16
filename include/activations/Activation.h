//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_ACTIVATION_H
#define CORTESIAN_ACTIVATION_H

#include "../utils/MetaBase.h"
#include <eigen3/Eigen/Core>

class Activation : public MetaBase {
public:
  virtual ~Activation() = default;

  /**
   * Activates the input, turning Weight*Vector + Bias into a vector of size of
   * owning layer's neurons.
   * @param in (w*X + b) from current layer.
   * @return the activated vector.
   */
  virtual Eigen::VectorXd function(const Eigen::VectorXd &in) = 0;

  /**
   * Differentiates the vector with respect to "itself" in terms of the
   * backpropagation. Might need to override for vector valued functions, since
   * Hessians need to be handled separately.
   * @param in vector to be differentiated.
   * @return differentiated vector.
   */
  virtual Eigen::VectorXd derivative(const Eigen::VectorXd &in) = 0;

  /**
   * To handle backpropagation, we take the derivative of the activation
   * function with respect to the gradient, forcing us not to derive the vector
   * against itself, but on the output of the gradient, i.e. dO/dGradient =
   * output*derivative(input) by the chain rule.
   * @param in activations of layers
   * @param out error terms (recursion gradients)
   * @return the hessian
   */
  virtual Eigen::MatrixXd derivative_on_input(const Eigen::VectorXd &in,
                                              const Eigen::VectorXd &out) {
    auto arrPred = out.array();
    auto arrRaw = in.array();
    auto diffRaw = derivative(arrRaw);
    return arrPred * diffRaw.array();
  };
};

#endif // CORTESIAN_ACTIVATION_H
