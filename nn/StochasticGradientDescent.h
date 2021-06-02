//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_STOCHASTICGRADIENTDESCENT_H
#define CORTESIAN_STOCHASTICGRADIENTDESCENT_H

#include <Eigen/Core>

#include "Optimizer.h"

class StochasticGradientDescent : public Optimizer {
 private:
  double l_r;

  void sgd(Eigen::MatrixXd &param, const Eigen::MatrixXd &delta_param) const;
  void sgd(Eigen::VectorXd &param, const Eigen::VectorXd &delta_param) const;

 public:
  explicit StochasticGradientDescent(double l_r = 0.0001) : l_r(l_r){};

  void change_weight(int layer_index, Eigen::MatrixXd &w,
                     const Eigen::MatrixXd &d_w) override;

  void change_bias(int layer_index, Eigen::VectorXd &b,
                   const Eigen::VectorXd &d_b) override;

  void initialize_optimizer(int layers, Eigen::MatrixXd w_seed,
                            Eigen::VectorXd b_seed) override;
};

#endif  // CORTESIAN_STOCHASTICGRADIENTDESCENT_H
