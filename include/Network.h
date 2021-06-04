//
// Created by Edwin Carlsson on 2021-05-31.
//

#ifndef CORTESIAN_NETWORK_H
#define CORTESIAN_NETWORK_H

#include "utils/MetaBase.h"
#include "utils/common.h"
#include <effolkronium/random.hpp>

using Random = effolkronium::random_static;

/**
 * A multi/single layer backpropagation feedforward network.
 */
class Network : MetaBase {
private:
  /**
   * Clipping is a struct representing if we should clip gradients or not.
   */
  struct Clipping : public MetaBase {
    bool clipping{false};
    double clip_factor{0.0};

  public:
    Clipping(bool should_clip, double clipping)
        : clip_factor(clipping), clipping(should_clip) {
      this->operator()("clipping", std::to_string(clip_factor));
      this->operator()("isClipping", should_clip ? "true" : "false");
    };

    Clipping() : clipping(false), clip_factor(0.0) {
      this->operator()("clipping", std::to_string(0.0));
      this->operator()("isClipping", "false");
    };
  };

protected:
  // Vector of abstract layers, core functionality
  std::vector<Layer *> m_layers;

  // virtual loss function, gradient and loss calculations
  LossFunction *m_loss;

  // Vector of evaluation funcions, evaluates the network
  std::vector<EvaluationFunction *> m_eval;

  // virtual Optimizer, optimizes weights/biases
  Optimizer *m_optimizer;

  // Clipping struct.
  Clipping m_clipping;

  // virtual initializer, provides weights/biases to network.
  ParameterInitializer *m_initializer;

  // These are private helper methods, but also the core of this class.
  void optimize();
  void back_propagate(const Eigen::VectorXd &matrix);
  void evaluate_for_back_prop(const DataSplit::DataPoint &point);
  void evaluate_for_back_prop(const DataSplit::DataSet &ds);

public:
  static std::string to_json(const Network &network);
  static Network from_json_file(const std::string& json_string);
  static Network from_json(const std::string& pre_loaded_json);

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

  void save(const std::string &file_name) const;

  void save(const std::string &&file_name) { save(file_name); };
};

#endif // CORTESIAN_NETWORK_H
