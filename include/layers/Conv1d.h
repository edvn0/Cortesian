//
// Created by Edwin Carlsson on 2021-06-03.
//

#ifndef CORTESIAN_CONV1D_H
#define CORTESIAN_CONV1D_H

#include "Layer.h"
class Conv1D : public Layer {
private:
  Eigen::VectorXd kernel;
  int m_kernel_size;

public:
  Conv1D(Activation *activation, int neurons, double reg = 0.0,
         int filter_size = 3)
      : Layer(activation, neurons, reg), m_kernel_size(3){};

public:
  Eigen::VectorXd calculate(const Eigen::VectorXd &in) override;

  void add_deltas(const Eigen::MatrixXd &d_w,
                  const Eigen::VectorXd &d_b) override;

  void fit(Optimizer *optimizer) override;

  Eigen::VectorXd activate(Eigen::VectorXd in, Eigen::VectorXd out) override;

  Eigen::MatrixXd error_derivative(const Eigen::VectorXd &previous,
                                   const Eigen::MatrixXd &current) override;

  Eigen::VectorXd calculate(Eigen::VectorXd &in) override;

  Eigen::MatrixXd get_weight() override;

  Eigen::MatrixXd get_activated() override;

  void set_params(Eigen::MatrixXd weight, Eigen::VectorXd bias) override;

  void set_delta_params(Eigen::MatrixXd d_weight,
                        Eigen::VectorXd d_bias) override;
};

#endif // CORTESIAN_CONV1D_H
