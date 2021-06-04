//
// Created by Edwin Carlsson on 2021-06-02.
//

#ifndef CORTESIAN_LAYER_H
#define CORTESIAN_LAYER_H

#include "../activations/Activation.h"
#include "../optimizers/Optimizer.h"
#include <eigen3/Eigen/Core>

/**
 * A layer is a virtual class handling feeding the network forward.
 * See {@link #Dense} for an implementation of this class.
 *
 * A layer is also in charge of updating the weights in back propagation,
 * however, instead of handling the logic of the updates itself, provides that
 * responsibility to {@link #Optimizer}.
 */
class Layer : public MetaBase {
protected:
  Layer(Activation *activation, int neurons, double l2)
      : m_activation(activation), m_neurons(neurons), m_l2(l2) {
    this->operator()("activation", activation->operator[]("activation"));
    this->operator()("neurons", std::to_string(neurons));
    this->operator()("l2", std::to_string(l2));
  };

  int m_layer_index;

  struct L2Tensors {
    Eigen::VectorXd m_l2Vector;
    Eigen::MatrixXd m_l2Matrix;
    bool are_created{false};
  };

  Activation *m_activation;
  L2Tensors regularization;
  double m_l2{0.0};

public:
  int m_neurons;

public:
  virtual ~Layer() = default;

  virtual Eigen::VectorXd calculate(const Eigen::VectorXd &in) = 0;

  virtual void add_deltas(const Eigen::MatrixXd &d_w,
                          const Eigen::VectorXd &d_b) = 0;

  virtual void fit(Optimizer *optimizer) = 0;

  virtual Eigen::VectorXd activate(Eigen::VectorXd in, Eigen::VectorXd out) = 0;

  virtual Eigen::MatrixXd error_derivative(const Eigen::VectorXd &previous,
                                           const Eigen::MatrixXd &current) = 0;

  virtual Eigen::VectorXd calculate(Eigen::VectorXd &in) = 0;

  virtual Eigen::MatrixXd get_weight() = 0;

  virtual Eigen::MatrixXd get_activated() = 0;

  virtual L2Tensors &get_regularization() { return regularization; };

  virtual void set_params(Eigen::MatrixXd weight, Eigen::VectorXd bias) = 0;

  virtual void set_delta_params(Eigen::MatrixXd d_weight,
                                Eigen::VectorXd d_bias) = 0;

  virtual int get_neurons() { return m_neurons; };

  virtual void set_index(int layer_index) {
    m_layer_index = layer_index;
    this->operator()("layer_index", std::to_string(m_layer_index));
  };

  virtual const double &get_l2() { return m_l2; }

  bool has_previous() { return m_layer_index != 0; };

  int get_layer_index() { return m_layer_index; }
};

#endif // CORTESIAN_LAYER_H
