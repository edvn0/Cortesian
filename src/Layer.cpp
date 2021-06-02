//
// Created by Edwin Carlsson on 2021-05-31.
//

#include "../include/Layer.h"

#include <iostream>
#include <utility>

Eigen::VectorXd Layer::activate(Eigen::VectorXd in, Eigen::VectorXd out) {
  return f_l->derivativeOnInput(std::move(in), std::move(out));
}

void Layer::fit(Optimizer *optimizer) {
  if (!has_previous()) return;

  if (deltas_added > 0) {
    if (l2 > 0) {
      // regularization
      Eigen::MatrixXd reg_weights =
          regularization.m_l2Matrix.array() * m_delta_weight.array();
      Eigen::VectorXd req_bias =
          regularization.m_l2Vector.array() * m_delta_bias.array();
      m_delta_weight -= reg_weights;
      m_delta_bias -= req_bias;
    }

    Eigen::MatrixXd average_delta_w =
        m_delta_weight.array() / (float)deltas_added;
    Eigen::VectorXd average_delta_b =
        m_delta_bias.array() / (float)deltas_added;
    optimizer->change_bias((int)m_index, m_bias, average_delta_b);
    optimizer->change_weight((int)m_index, m_weight, average_delta_w);
    m_delta_weight = m_delta_weight.setZero();
    m_delta_bias = m_delta_bias.setZero();
    deltas_added = 0;
  }
}

void Layer::add_deltas(const Eigen::MatrixXd &d_w, const Eigen::VectorXd &d_b) {
  m_delta_weight += d_w;
  m_delta_bias += d_b;
  deltas_added = deltas_added + 1;
}

Eigen::VectorXd Layer::calculate(const Eigen::VectorXd &in) {
  if (!has_previous()) {
    activated = in;
  } else {
    auto out = f_l->function(m_weight * in + m_bias);
    activated = out;
  }
  return activated;
}

Layer::Layer(int neurons, double l2, Activation *activation, Eigen::MatrixXd ws,
             Eigen::VectorXd bs)
    : neurons(neurons),
      l2(l2),
      f_l(activation),
      m_delta_weight(std::move(ws)),
      m_delta_bias(std::move(bs)) {}

Layer::Layer(Activation *activation, int neurons)
    : f_l(activation), neurons(neurons), m_previous(nullptr) {}

Layer::Layer(Activation *activation, int neurons, double l2)
    : f_l(activation), neurons(neurons), m_previous(nullptr), l2(l2) {}

std::ostream &operator<<(std::ostream &os, const Layer &layer) {
  os << "m_weight: " << layer.m_weight.rows() << " X " << layer.m_weight.cols()
     << "\nm_bias: " << layer.m_bias.rows() << " X 1"
     << "\nneurons: " << layer.neurons << "\n";
  return os;
}

void Layer::set_delta_params(Eigen::MatrixXd d_weight, Eigen::VectorXd d_bias) {
  m_delta_weight = std::move(d_weight);
  m_delta_bias = std::move(d_bias);
  deltas_added = 0;
}

void Layer::set_params(Eigen::MatrixXd weight, Eigen::VectorXd bias) {
  m_weight = std::move(weight);
  m_bias = std::move(bias);
}

Eigen::MatrixXd Layer::error_derivative(const Eigen::VectorXd &prev_activation,
                                        const Eigen::MatrixXd &current) {
  return f_l->derivativeOnInput(prev_activation, current);
}
Eigen::MatrixXd Layer::previous_activation() {
  return m_previous->get_activated().transpose();
}

Layer *Layer::get_previous() { return m_previous; }

Layer::Layer(const Layer &other) {
  m_index = other.m_index;
  m_weight = other.m_weight;
  m_delta_weight = other.m_delta_weight;
  m_delta_bias = other.m_delta_bias;
  m_bias = other.m_bias;
  activated = other.activated;
  neurons = other.neurons;
  m_previous = other.m_previous;
  l2 = other.l2;
  f_l = other.f_l;
}

void Layer::set_index(size_t i) { m_index = i; }
