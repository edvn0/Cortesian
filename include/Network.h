//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_NETWORK_H
#define CORTESIAN_NETWORK_H

#include "utils/common.h"

class Network {
private:
  struct Clipping {
    bool clipping{false};
    double clip_factor{0.0};
  };

protected:
  std::vector<Layer *> m_layers;
  LossFunction *m_loss;
  std::vector<EvaluationFunction *> m_eval;
  Optimizer *m_optimizer;
  Clipping m_clipping;
  ParameterInitializer *m_initializer;

  // These are private helper methods, but also the core of this class.
  void optimize();
  void back_propagate(const Eigen::VectorXd &matrix);
  void evaluate_for_back_prop(const DataSplit::DataPoint &point);
  void evaluate_for_back_prop(const DataSplit::DataSet &ds);

public:
  Network();
  explicit Network(NetworkBuilder builder);
  explicit Network(NetworkBuilder &&builder) : Network(builder){};
  ~Network();

  virtual BackPropStatistics fit(const std::vector<Eigen::VectorXd> &X,
                                 const std::vector<Eigen::VectorXd> &Y,
                                 int epochs, int batch_size, double train_split,
                                 bool should_shuffle_training);

  BackPropStatistics fit_tensor(Eigen::MatrixXd &X, Eigen::MatrixXd &Y,
                                int epochs, int batch_size,
                                Eigen::MatrixXd &X_validate,
                                Eigen::MatrixXd &Y_validate);

  Eigen::VectorXd predict(const Eigen::VectorXd &vector);

  std::vector<Eigen::VectorXd> evaluate(const std::vector<Eigen::VectorXd> &Xs);

  std::vector<Eigen::VectorXd> evaluate(const Eigen::MatrixXd &Xs);

  friend std::ostream &operator<<(std::ostream &os, const Network &network);

  Eigen::VectorXd classify(const Eigen::VectorXd &vector);

  static std::vector<DataSplit::DataSet>
  generate_splits(Eigen::MatrixXd &X_tensor, Eigen::MatrixXd &Y_tensor,
                  size_t from_index, size_t to_index, int i);

  static void print_epoch_information(double loss,
                                      const std::vector<double> &metrics);
};

#endif // CORTESIAN_NETWORK_H
