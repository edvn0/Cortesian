//
// Created by Edwin Carlsson on 2021-05-31.
//

#include "../../include/layers/Dense.h"
#include "../../include/utils/DataReader.h"

#include <iostream>
#include <utility>

Eigen::VectorXd Dense::activate(Eigen::VectorXd in, Eigen::VectorXd out) {
  return m_activation->derivative_on_input(std::move(in), std::move(out));
}

void Dense::fit(Optimizer *optimizer) {
  if (!has_previous())
    return;

  if (deltas_added > 0) {
    if (m_l2 > 0) {
      m_delta_weight.unaryExpr([=](double t) { return t - t * m_l2; });
      m_delta_bias.unaryExpr([=](double t) { return t - t * m_l2; });
    }

    Eigen::MatrixXd average_delta_w =
        m_delta_weight.array() / (float)deltas_added;
    Eigen::VectorXd average_delta_b =
        m_delta_bias.array() / (float)deltas_added;

    optimizer->change_weight((int)m_layer_index, m_weight, average_delta_w);
    optimizer->change_bias((int)m_layer_index, m_bias, average_delta_b);
    m_delta_weight =
        Eigen::MatrixXd::Zero(m_delta_weight.rows(), m_delta_weight.cols());
    m_delta_bias = Eigen::VectorXd::Zero(m_delta_bias.rows());
  }
  deltas_added = 0;
}

void Dense::add_deltas(const Eigen::MatrixXd &d_w, const Eigen::VectorXd &d_b) {
  m_delta_weight += d_w;
  m_delta_bias += d_b;
  deltas_added = deltas_added + 1;
}

Dense::Dense(int neurons, double l2, Activation *activation, Eigen::MatrixXd ws,
             Eigen::VectorXd bs)
    : Layer(activation, neurons, l2), m_delta_weight(std::move(ws)),
      m_delta_bias(std::move(bs)) {}

Dense::Dense(Activation *activation, int neurons)
    : Layer(activation, neurons, 0.0) {}

Dense::Dense(Activation *activation, int neurons, double l2)
    : Layer(activation, neurons, l2) {}

std::ostream &operator<<(std::ostream &os, const Dense &layer) {
  os << "m_weight: " << layer.m_weight.rows() << " X " << layer.m_weight.cols()
     << "\nm_bias: " << layer.m_bias.rows() << " X 1"
     << "\nneurons: " << layer.m_neurons << "\n";
  return os;
}

void Dense::set_delta_params(Eigen::MatrixXd d_weight, Eigen::VectorXd d_bias) {
  m_delta_weight = std::move(d_weight);
  m_delta_bias = std::move(d_bias);
  deltas_added = 0;
}

void Dense::set_params(Eigen::MatrixXd weight, Eigen::VectorXd bias) {
  m_weight = std::move(weight);
  m_bias = std::move(bias);
}

Eigen::MatrixXd Dense::error_derivative(const Eigen::VectorXd &prev_activation,
                                        const Eigen::MatrixXd &current) {
  return m_activation->derivative_on_input(prev_activation, current);
}

Dense::Dense(const Dense &other) : Layer(other) {
  m_layer_index = other.m_layer_index;
  m_weight = other.m_weight;
  m_delta_weight = other.m_delta_weight;
  m_delta_bias = other.m_delta_bias;
  m_bias = other.m_bias;
  activated = other.activated;
  m_neurons = other.m_neurons;
  m_l2 = other.m_l2;
  m_activation = other.m_activation;
}

Dense::L2Tensors &Dense::get_regularization() { return regularization; }

Eigen::VectorXd Dense::calculate(Eigen::VectorXd &in) {
  if (!has_previous()) {
    activated = in;
  } else {
    auto out = m_activation->function(m_weight * in + m_bias);
    activated = out;
  }
  return activated;
}

Eigen::VectorXd Dense::calculate(const Eigen::VectorXd &in) {
  if (!has_previous()) {
    activated = in;
  } else {
    activated = m_activation->function(m_weight * in + m_bias);
  }
  return activated;
}

void Dense::serialize_parameters() {
  this->operator()("weight", eigen_to_json(m_weight));
  this->operator()("bias", eigen_to_json(m_bias));
}
