//
// Created by Edwin Carlsson on 2021-06-02.
//

#ifndef CORTESIAN_ADAM_H
#define CORTESIAN_ADAM_H

#include "Optimizer.h"
#include <vector>

class Adam : public Optimizer {
private:
  std::vector<Eigen::MatrixXd> w_m;
  std::vector<Eigen::MatrixXd> w_n;
  std::vector<Eigen::VectorXd> b_m;
  std::vector<Eigen::VectorXd> b_n;

  std::vector<bool> bias_initialized;
  std::vector<bool> weight_initialized;

  const double learning_rate{0.01};
  const double beta_one{0.9};
  const double beta_two{0.999};

  static constexpr double epsilon = 1e-8;

public:
  explicit Adam(double learning, double alpha_one = 0.9,
                double alpha_two = 0.999)
      : learning_rate(learning), beta_one(alpha_one), beta_two(alpha_two) {
    this->operator()("optimizer", "Adam");
    this->operator()("learning", std::to_string(learning));
    this->operator()("alpha_one", std::to_string(alpha_one));
    this->operator()("alpha_two", std::to_string(alpha_two));
  }

  void change_weight(int layer_index, Eigen::MatrixXd &w,
                     const Eigen::MatrixXd &d_w) override;

  void change_bias(int layer_index, Eigen::VectorXd &b,
                   const Eigen::VectorXd &d_b) override;

  void initialize_optimizer(int layers, Eigen::MatrixXd w_seed,
                            Eigen::VectorXd b_seed) override;
};

#endif // CORTESIAN_ADAM_H
