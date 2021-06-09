//
// Created by Edwin Carlsson on 2021-06-02.
//

#include "../../include/optimizers/Adam.h"
#include <iostream>

void Adam::change_weight(int layer_index, Eigen::MatrixXd &w,
                         const Eigen::MatrixXd &d_w) {
  int exponent = layer_index + 1;
  Eigen::MatrixXd m_hat;
  Eigen::MatrixXd v_hat;
  if (!weight_initialized[layer_index]) {
    weight_initialized[layer_index] = true;

    w_m[layer_index] = d_w.array() * (1 - beta_one);
    w_n[layer_index] = d_w.array() * d_w.array() * (1 - beta_two);
  } else {
    Eigen::MatrixXd m =
        (w_m[layer_index] * beta_one).array() + d_w.array() * (1 - beta_one);
    Eigen::MatrixXd v = (w_n[layer_index] * beta_two).array() +
                        d_w.array() * d_w.array() * (1 - beta_two);

    w_m[layer_index] = m;
    w_n[layer_index] = v;
  }

  m_hat = w_m[layer_index].array() / (1 - pow(beta_one, exponent));
  v_hat = w_n[layer_index].array() / (1 - pow(beta_two, exponent));

  auto denom = v_hat.array().sqrt() + epsilon;
  auto num = m_hat * learning_rate;
  Eigen::MatrixXd adam = num.array() / denom.array();
  w -= adam;
}

void Adam::change_bias(int layer_index, Eigen::VectorXd &b,
                       const Eigen::VectorXd &d_b) {
  int exponent = layer_index + 1;
  Eigen::MatrixXd m_hat;
  Eigen::MatrixXd v_hat;
  if (!bias_initialized[layer_index]) {
    bias_initialized[layer_index] = true;

    b_m[layer_index] = d_b.array() * (1 - beta_one);
    b_n[layer_index] = d_b.array() * d_b.array() * (1 - beta_two);
  } else {
    Eigen::MatrixXd m =
        (b_m[layer_index] * beta_one).array() + d_b.array() * (1 - beta_one);
    Eigen::MatrixXd v = (b_n[layer_index] * beta_two).array() +
                        d_b.array() * d_b.array() * (1 - beta_two);

    b_m[layer_index] = m;
    b_n[layer_index] = v;
  }

  m_hat = b_m[layer_index].array() / (1 - pow(beta_one, exponent));
  v_hat = b_n[layer_index].array() / (1 - pow(beta_two, exponent));

  auto denom = v_hat.array().sqrt() + epsilon;
  auto num = m_hat * learning_rate;
  Eigen::VectorXd adam = num.array() / denom.array();
  b = b - adam;
}

void Adam::initialize_optimizer(int layers, Eigen::MatrixXd w_seed,
                                Eigen::VectorXd b_seed) {
  w_m.reserve(layers);
  b_m.reserve(layers);
  w_n.reserve(layers);
  b_n.reserve(layers);
  bias_initialized.reserve(layers);
  weight_initialized.reserve(layers);
  for (size_t i = 0; i < layers; i++) {
    w_m.emplace_back(Eigen::MatrixXd::Zero(w_seed.rows(), w_seed.cols()));
    w_n.emplace_back(Eigen::MatrixXd::Zero(w_seed.rows(), w_seed.cols()));
    b_m.emplace_back(Eigen::VectorXd::Zero(b_seed.rows()));
    b_n.emplace_back(Eigen::VectorXd::Zero(b_seed.rows()));
    bias_initialized.emplace_back(false);
    weight_initialized.emplace_back(false);
  }
}
