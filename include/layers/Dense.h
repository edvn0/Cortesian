//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_DENSE_H
#define CORTESIAN_DENSE_H

#include <eigen3/Eigen/Core>
#include <ostream>

#include "../activations/Activation.h"
#include "../optimizers/Optimizer.h"
#include "Layer.h"

class Dense : public Layer {
private:
  Eigen::MatrixXd m_weight;
  Eigen::VectorXd m_bias;
  Eigen::MatrixXd m_delta_weight;
  Eigen::VectorXd m_delta_bias;

  Eigen::VectorXd activated;
  int deltas_added{0};

public:
  Dense(Activation *activation, int neurons, double l2);

  Dense(Activation *activation, int neurons);

  Dense(const Dense &other);

  Dense(int neurons, double l2, Activation *activation, Eigen::MatrixXd ws,
        Eigen::VectorXd bs);

  ~Dense() = default;

  void add_deltas(const Eigen::MatrixXd &d_w,
                  const Eigen::VectorXd &d_b) override;

  void fit(Optimizer *optimizer) override;

  Eigen::VectorXd activate(Eigen::VectorXd in, Eigen::VectorXd out) override;

  Eigen::MatrixXd error_derivative(const Eigen::VectorXd &previous,
                                   const Eigen::MatrixXd &current) override;

  Eigen::VectorXd calculate(Eigen::VectorXd &in) override;

public:
  Eigen::VectorXd get_bias() { return m_bias; };

  Eigen::MatrixXd get_weight() override { return m_weight; };

  Eigen::MatrixXd get_activated() override { return activated; };

  void set_params(Eigen::MatrixXd weight, Eigen::VectorXd bias) override;

  void set_delta_params(Eigen::MatrixXd d_weight,
                        Eigen::VectorXd d_bias) override;

  L2Tensors &get_regularization() override;

  friend std::ostream &operator<<(std::ostream &os, const Dense &layer);
  Eigen::VectorXd calculate(const Eigen::VectorXd &in) override;
};

#endif // CORTESIAN_DENSE_H
