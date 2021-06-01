//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_NETWORKBUILDER_H
#define CORTESIAN_NETWORKBUILDER_H

#include "EvaluationFunction.h"
#include "Layer.h"
#include "LossFunction.h"
#include "Optimizer.h"
#include "ParameterInitializer.h"
#include <vector>

class NetworkBuilder {
private:
  double m_gradient_clipping;
  int m_total;
  std::vector<Layer> m_layers;

  Optimizer *m_optimizer;
  ParameterInitializer *m_initializer;
  LossFunction *m_loss_function;
  std::vector<EvaluationFunction *> m_evaluation_function{};

public:
  NetworkBuilder() = default;

  NetworkBuilder(const NetworkBuilder &other) {
    m_initializer = other.m_initializer;
    m_loss_function = other.m_loss_function;
    m_evaluation_function = other.m_evaluation_function;
    m_optimizer = other.m_optimizer;
    m_layers = other.m_layers;
    m_gradient_clipping = other.m_gradient_clipping;
    m_total = other.m_total;
  }

  NetworkBuilder &clipping(double clip_gradients) {
    m_gradient_clipping = clip_gradients;
    return *this;
  }

  NetworkBuilder &evaluation_function(EvaluationFunction *evaluation_function) {
    m_evaluation_function.emplace_back(evaluation_function);
    return *this;
  }

  NetworkBuilder &
  evaluation_function(std::initializer_list<EvaluationFunction *> functions) {
    m_evaluation_function = functions;
    return *this;
  }

  NetworkBuilder &initializer(ParameterInitializer *initializer) {
    m_initializer = initializer;
    return *this;
  }

  NetworkBuilder &loss_function(LossFunction *loss_function) {
    m_loss_function = loss_function;
    return *this;
  }

  NetworkBuilder &optimizer(Optimizer *optimizer) {
    m_optimizer = optimizer;
    return *this;
  }

  NetworkBuilder &layer(Layer &&layer) {
    m_layers.emplace_back(layer);
    return *this;
  }

  std::vector<int> compile() {
    int layers = m_layers.size();
    m_total = layers;
    std::vector<int> sizes;
    sizes.reserve(m_total);
    for (int i = 0; i < m_total; ++i) {
      sizes.push_back(m_layers[i].get_neurons());
    }
    return sizes;
  }

  [[nodiscard]] size_t get_total() const { return m_total; }
  LossFunction *get_loss() { return m_loss_function; }
  std::vector<EvaluationFunction *> get_eval() { return m_evaluation_function; }
  Optimizer *get_optimizer() { return m_optimizer; }
  ParameterInitializer *get_initializer() { return m_initializer; }
  [[nodiscard]] bool should_clip() const { return m_gradient_clipping > 0; };
  [[nodiscard]] double clip_factor() const { return m_gradient_clipping; }
  std::vector<Layer> get_layers() { return m_layers; }
};

#endif // CORTESIAN_NETWORKBUILDER_H
