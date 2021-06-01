//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_OPTIMIZER_H
#define CORTESIAN_OPTIMIZER_H

#include "../libs/Eigen/Core"

class Optimizer {
public:
  virtual ~Optimizer() = default;

public:
  virtual void change_weight(int layer_index, Eigen::MatrixXd& w,
                                        const Eigen::MatrixXd& d_w) = 0;

  virtual void change_bias(int layer_index, Eigen::VectorXd& b,
                                      const Eigen::VectorXd& d_b) = 0;

  virtual void initialize_optimizer(int layers, Eigen::MatrixXd w_seed,
                                    Eigen::VectorXd b_seed) = 0;
};

#endif // CORTESIAN_OPTIMIZER_H
