//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_NETWORK_H
#define CORTESIAN_NETWORK_H

#include "common.h"

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

  // These are private helper methods, but also the core of this class.
  void optimize();
  void back_propagate(const Eigen::VectorXd &matrix);

public:
  Network();
  explicit Network(NetworkBuilder builder);
  explicit Network(NetworkBuilder &&builder) : Network(builder){};
  ~Network() = default;

  BackPropStatistics fit(const std::vector<Eigen::VectorXd> &X,
                         const std::vector<Eigen::VectorXd> &Y, int epochs,
                         int batch_size, double train_split = 0.8);

  Eigen::MatrixXd evaluate(const DataSplit::DataPoint &point);

  Eigen::VectorXd predict(const Eigen::VectorXd &vector);

  std::vector<Eigen::VectorXd> evaluate(const std::vector<Eigen::VectorXd> &Xs);

  friend std::ostream &operator<<(std::ostream &os, const Network &network);

  Eigen::VectorXd classify(const Eigen::VectorXd &vector);
};

#endif // CORTESIAN_NETWORK_H
