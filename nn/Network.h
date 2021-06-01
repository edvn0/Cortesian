//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_NETWORK_H
#define CORTESIAN_NETWORK_H

#include "BackPropStatistics.h"
#include "BlockTimer.h"
#include "DataSplit.h"
#include "EvaluationFunction.h"
#include "Layer.h"
#include "LossFunction.h"
#include "NetworkBuilder.h"
#include "ParameterInitializer.h"
#include "MathUtils.h"
#include <iostream>
#include <ostream>
#include <vector>

class Network {
private:
  struct Clipping {
    bool clipping{false};
    double clip_factor{0.0};
  };

private:
  std::vector<Layer> m_layers;
  LossFunction *m_loss;
  std::vector<EvaluationFunction *> m_eval;
  Optimizer *m_optimizer;
  Clipping m_clipping;
  ParameterInitializer *m_initializer;

  void optimize();
  void back_propagate(const Eigen::VectorXd &matrix);

public:
  Network();
  explicit Network(NetworkBuilder builder);
  ~Network() = default;

  BackPropStatistics fit(const std::vector<Eigen::VectorXd> &X,
                         const std::vector<Eigen::VectorXd> &Y, int epochs,
                         int batch_size);

  Eigen::MatrixXd evaluate(const DataSplit::DataPoint &point);

  Eigen::VectorXd predict(const Eigen::VectorXd &vector);

  std::vector<Eigen::VectorXd> evaluate(const std::vector<Eigen::VectorXd> &Xs);

  friend std::ostream &operator<<(std::ostream &os, const Network &network);

  Eigen::VectorXd classify(const Eigen::VectorXd& vector);
};

#endif // CORTESIAN_NETWORK_H
