//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_LAYER_H
#define CORTESIAN_LAYER_H

#include "../libs/Eigen/Core"
#include "Activation.h"
#include "Optimizer.h"
#include <ostream>

class Layer {
private:
  Eigen::MatrixXd m_weight;
  Eigen::VectorXd m_bias;
  Eigen::MatrixXd m_delta_weight;
  Eigen::VectorXd m_delta_bias;

  int neurons;

  Layer *m_previous;
  Activation *f_l;

  Eigen::VectorXd activated;

  size_t m_index;
  int deltas_added{0};

  struct L2Tensors {
    Eigen::VectorXd m_l2Vector;
    Eigen::MatrixXd m_l2Matrix;
    bool are_created{false};
  };

public:
  Layer(Activation *activation, int neurons, double l2);

  Layer(Activation *activation, int neurons);

  Layer(const Layer &other);

  Layer(int neurons, double l2, Activation *activation, Eigen::MatrixXd ws,
        Eigen::VectorXd bs);

  ~Layer() = default;

  Eigen::VectorXd calculate(const Eigen::VectorXd &in);

  void add_deltas(const Eigen::MatrixXd &d_w, const Eigen::VectorXd &d_b);

  void fit(Optimizer *optimizer);

  Eigen::VectorXd activate(Eigen::VectorXd in, Eigen::VectorXd out);

  L2Tensors regularization;

  double l2{0.0};

public:
  /**
   * Returns true if the previous layer is not nullptr.
   * @return Returns true if the previous layer is not nullptr.
   */
  bool has_previous() { return m_previous != nullptr; };

  [[nodiscard]] int get_neurons() const { return neurons; };

  Eigen::VectorXd get_bias() { return m_bias; };

  /**
   * Returns the transpose of the previous layer's activation.
   * Utility for back propagation.
   * @return transpose of prev layer activation.
   */
  Eigen::MatrixXd previous_activation();

  Eigen::MatrixXd get_weight() { return m_weight; };

  Eigen::MatrixXd get_activated() { return activated; };

  void set_previous(Layer *prev) { m_previous = prev; }

  Layer *get_previous();

  void set_params(Eigen::MatrixXd weight, Eigen::VectorXd bias);

  void set_delta_params(Eigen::MatrixXd d_weight, Eigen::VectorXd d_bias);

  Eigen::MatrixXd error_derivative(const Eigen::VectorXd &previous,
                                   const Eigen::MatrixXd &current);

  void set_index(size_t i);

  friend std::ostream &operator<<(std::ostream &os, const Layer &layer);
};

#endif // CORTESIAN_LAYER_H
