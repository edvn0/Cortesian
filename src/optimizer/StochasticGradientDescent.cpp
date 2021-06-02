//
// Created by Edwin Carlsson on 2021-05-31.
//

#include "../../include/optimizers/StochasticGradientDescent.h"

void StochasticGradientDescent::change_weight(int layer_index,
                                              Eigen::MatrixXd &w,
                                              const Eigen::MatrixXd &d_w) {
  sgd(w, d_w);
}

void StochasticGradientDescent::change_bias(int layer_index, Eigen::VectorXd &b,
                                            const Eigen::VectorXd &d_b) {
  sgd(b, d_b);
}

void StochasticGradientDescent::initialize_optimizer(int layers,
                                                     Eigen::MatrixXd w_seed,
                                                     Eigen::VectorXd b_seed) {}

void StochasticGradientDescent::sgd(Eigen::MatrixXd &param,
                                    const Eigen::MatrixXd &delta_param) const {
  param -= l_r * delta_param;
}

void StochasticGradientDescent::sgd(Eigen::VectorXd &param,
                                    const Eigen::VectorXd &delta_param) const {
  param -= l_r * delta_param;
}