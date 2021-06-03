//
// Created by Edwin Carlsson on 2021-05-31.
//

#include "../include/Network.h"

Network::Network()
    : m_loss(nullptr), m_eval({nullptr}), m_optimizer(nullptr),
      m_initializer(nullptr) {}

Network::Network(NetworkBuilder builder) {
  if (builder.is_valid() != NetworkBuilder::Validity::VALID) {
    throw std::runtime_error(
        NetworkBuilder::validity_to_string(builder.is_valid()));
  }

  m_loss = builder.get_loss();
  m_eval = builder.get_eval();
  m_optimizer = builder.get_optimizer();
  m_initializer = builder.get_initializer();
  m_clipping = {builder.should_clip(), builder.clip_factor()};

  m_layers = builder.get_layers();
  //  m_layers[0].set_activation_function(DoNothingFunction());
  m_layers[0]->set_index(0);

  auto structure = builder.compile();
  m_initializer->init(structure);
  auto ws = m_initializer->get_weight_params();
  auto bs = m_initializer->get_bias_params();
  auto dws = m_initializer->get_delta_weight_params();
  auto dbs = m_initializer->get_delta_bias_params();

  m_optimizer->initialize_optimizer((int)m_layers.size(), {}, {});

  for (size_t i = 1; i < builder.get_total(); i++) {
    Layer *new_layer = m_layers[i];
    auto layer_weight = ws[i - 1];
    auto layer_bias = bs[i - 1];
    auto layer_delta_weight = dws[i - 1];
    auto layer_delta_bias = dbs[i - 1];

    new_layer->set_params(layer_weight, layer_bias);
    new_layer->set_delta_params(layer_delta_weight, layer_delta_bias);
    new_layer->set_index(i);

    auto &reg = new_layer->get_regularization();
    if (!reg.are_created) {
      reg.m_l2Matrix = Eigen::MatrixXd::Constant(
          layer_weight.rows(), layer_weight.cols(), new_layer->get_l2());
      reg.m_l2Vector =
          Eigen::VectorXd::Constant(layer_weight.rows(), new_layer->get_l2());
      reg.are_created = true;
    }

    new_layer->set_previous(m_layers[i - 1]);

    m_layers[i] = new_layer;
  }
}

std::vector<Eigen::VectorXd>
Network::evaluate(const std::vector<Eigen::VectorXd> &Xs) {
  std::vector<Eigen::VectorXd> predicted;
  predicted.reserve(Xs.size());

  for (const auto &x : Xs) {
    predicted.emplace_back(predict(x));
  }

  return predicted;
}

std::ostream &operator<<(std::ostream &os, const Network &network) {
  os << "\nOptimizer:" << network.m_optimizer << "\nLoss: " << network.m_loss
     << "\nInitializer: " << network.m_initializer << "\nEvaluators: [";
  for (auto *eval : network.m_eval) {
    os << eval << ", ";
  }
  os << "]\n";
  return os;
}

void Network::optimize() {
  for (auto &layer : m_layers) {
    layer->fit(m_optimizer);
  }
}

void Network::evaluate_for_back_prop(const DataSplit::DataPoint &point) {
  auto Xs = point.X;
  auto Ys = point.Y;

  for (auto &layer : m_layers) {
    Xs = layer->calculate(Xs);
  }

  back_propagate(Ys);
}

void Network::evaluate_for_back_prop(const DataSplit::DataSet &ds) {
  for (const auto &d : ds.rows) {
    evaluate_for_back_prop(d);
  }
}

std::vector<Eigen::VectorXd> Network::evaluate(const Eigen::MatrixXd &Xs) {
  size_t rows = Xs.rows();
  std::vector<Eigen::VectorXd> evaluated;
  evaluated.reserve(rows);
  for (long i = 0; i < rows; i++) {
    Eigen::VectorXd xs = Xs.row(i);

    for (auto &layer : m_layers) {
      xs = layer->calculate(xs);
    }

    evaluated.emplace_back(xs);
  }
  return evaluated;
}

void Network::back_propagate(const Eigen::VectorXd &real) {
  Layer *last_layer = m_layers.back();
  Eigen::VectorXd z = last_layer->get_activated();

  Eigen::MatrixXd gradient = m_loss->apply_loss_gradient(z, real);
  if (m_clipping.clipping) {
    double norm = gradient.norm();
    if (norm >= m_clipping.clip_factor) {
      gradient = m_clipping.clip_factor * gradient / norm;
    }
  }

  do {
    auto dc_di = last_layer->error_derivative(z, gradient);
    auto previous_activation = last_layer->previous_activation();

    Eigen::MatrixXd d_w = dc_di * previous_activation;

    last_layer->add_deltas(d_w, dc_di);

    gradient = last_layer->get_weight().transpose() * dc_di;

    last_layer = last_layer->get_previous();
    z = last_layer->get_activated();

  } while (last_layer->has_previous());
}

Eigen::VectorXd Network::predict(const Eigen::VectorXd &vector) {
  Eigen::VectorXd eval = vector;
  for (auto &layer : m_layers) {
    eval = layer->calculate(eval);
  }

  return eval;
}

Eigen::VectorXd Network::classify(const Eigen::VectorXd &vector) {
  auto out = predict(vector);
  Eigen::VectorXd vec(1);
  vec << arg_max(out);
  return vec;
}

BackPropStatistics Network::fit_tensor(Eigen::MatrixXd &X, Eigen::MatrixXd &Y,
                                       int epochs, int batch_size,
                                       Eigen::MatrixXd &X_validate,
                                       Eigen::MatrixXd &Y_validate) {

  BackPropStatistics backPropStatistics(m_eval.size());

  size_t end_size = X.rows();
  size_t start_size = 0;

  auto splits = generate_splits(X, Y, start_size, end_size, batch_size);

  for (int i = 0; i < epochs; i++) {
    BlockTimer t;
    for (auto &dps : splits) {
      evaluate_for_back_prop(dps);
      optimize();
    }
    t.stop();

    auto timeEpoch = t.elapsedSeconds();
    const std::vector<Eigen::VectorXd> evaluated = evaluate(X_validate);
    auto loss = m_loss->apply_loss(evaluated, Y_validate);
    std::vector<double> metrics;
    metrics.reserve(m_eval.size());
    for (auto *eval : m_eval) {
      metrics.emplace_back(eval->apply_evaluation(evaluated, Y_validate));
    }

    std::cout << "Loss: " << loss << ", Evaluation Scores: [";
    for (auto &val : metrics) {
      std::cout << val << ", ";
    }
    std::cout << "]\n";

    backPropStatistics.update(timeEpoch, loss, metrics);
  }
  backPropStatistics.finalize();
  return backPropStatistics;
}

BackPropStatistics Network::fit(const std::vector<Eigen::VectorXd> &X,
                                const std::vector<Eigen::VectorXd> &Y,
                                int epochs, int batch_size, double train_split,
                                bool should_shuffle_training) {
  auto training_size = (size_t)((int)X.size() * train_split);

  std::vector<Eigen::VectorXd> train_x(&X[0], &X[training_size]);
  std::vector<Eigen::VectorXd> train_y(&Y[0], &Y[training_size]);
  std::vector<Eigen::VectorXd> validate_x(&X[training_size], &X[X.size() - 1]);
  std::vector<Eigen::VectorXd> validate_y(&Y[training_size], &Y[Y.size() - 1]);

  DataSplit batch_split(batch_size, train_x, train_y);

  BackPropStatistics backPropStatistics(m_eval.size());

  for (int i = 0; i < epochs; i++) {
    BlockTimer t;
    auto splits = batch_split.get_splits(should_shuffle_training);
    for (auto &dps : splits) {
      evaluate_for_back_prop(dps);
      optimize();
    }
    t.stop();
    auto timeEpoch = t.elapsedSeconds();

    const std::vector<Eigen::VectorXd> evaluated = evaluate(validate_x);
    auto loss = m_loss->apply_loss(evaluated, validate_y);
    std::vector<double> metrics;
    metrics.reserve(m_eval.size());
    for (auto *eval : m_eval) {
      metrics.emplace_back(eval->apply_evaluation(evaluated, validate_y));
    }

    std::cout << "Loss: " << loss << ", Evaluation Scores: [";
    for (auto &val : metrics) {
      std::cout << val << ", ";
    }
    std::cout << "]\n";

    backPropStatistics.update(timeEpoch, loss, metrics);
  }
  backPropStatistics.finalize();
  return backPropStatistics;
}

std::vector<DataSplit::DataSet>
Network::generate_splits(Eigen::MatrixXd &X_tensor, Eigen::MatrixXd &Y_tensor,
                         size_t from_index, size_t to_index, int batch_size) {
  std::vector<DataSplit::DataSet> data_sets;
  data_sets.reserve((int)(X_tensor.size() / batch_size));
  for (size_t i = from_index; i < to_index; i += batch_size) {
    if (i >= to_index)
      break;

    std::vector<DataSplit::DataPoint> dps;
    dps.reserve(batch_size);
    size_t max_loop = std::min((i + batch_size), to_index);
    for (size_t j = i; j < max_loop; j++) {
      Eigen::VectorXd col_x = X_tensor.row((long)j);
      Eigen::VectorXd col_y = Y_tensor.row((long)j);
      dps.emplace_back(DataSplit::DataPoint{col_x, col_y});
    }
    DataSplit::DataSet ds;
    ds.rows = dps;
    data_sets.emplace_back(ds);
  }
  return data_sets;
}
Network::~Network() {
  std::cout << "deleting network.";
  delete m_optimizer;
  delete m_initializer;
  delete m_loss;

  for (auto eval : m_eval) {
    delete eval;
  }

  for (auto layer : m_layers) {
    delete layer;
  }

  m_eval.clear();
  m_layers.clear();
}
