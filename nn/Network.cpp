//
// Created by Edwin Carlsson on 2021-05-31.
//

#include "Network.h"
#include "BackPropStatistics.h"
#include "BlockTimer.h"
#include "DataSplit.h"

Network::Network()
    : m_loss(nullptr), m_eval({nullptr}), m_optimizer(nullptr),
      m_initializer(nullptr) {}

BackPropStatistics Network::fit(const std::vector<Eigen::VectorXd> &X,
                                const std::vector<Eigen::VectorXd> &Y,
                                int epochs, int batch_size) {
  DataSplit batch_split(batch_size, X, Y);

  BackPropStatistics backPropStatistics(m_eval.size());

  for (int i = 0; i < epochs; i++) {
    BlockTimer t;
    for (auto &dps : batch_split.get_splits()) {
      for (const auto &point : dps.rows) {
        evaluate(point);
      }
      optimize();
    }
    t.stop();
    auto timeEpoch = t.elapsedSeconds();

    const std::vector<Eigen::VectorXd> evaluated = evaluate(X);
    auto loss = m_loss->apply_loss(evaluated, Y);
    std::vector<double> metrics;
    metrics.reserve(m_eval.size());
    for (auto *eval : m_eval) {
      metrics.emplace_back(eval->apply_evaluation(evaluated, Y));
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

std::vector<Eigen::VectorXd>
Network::evaluate(const std::vector<Eigen::VectorXd> &Xs) {
  std::vector<Eigen::VectorXd> predicted;
  predicted.reserve(Xs.size());

  for (const auto &x : Xs) {
    predicted.emplace_back(predict(x));
  }

  return predicted;
}

Network::Network(NetworkBuilder builder) {
  m_loss = builder.get_loss();
  m_eval = builder.get_eval();
  m_optimizer = builder.get_optimizer();
  m_initializer = builder.get_initializer();
  m_clipping = {builder.should_clip(), builder.clip_factor()};

  m_layers = builder.get_layers();
  //  m_layers[0].set_activation_function(DoNothingFunction());
  m_layers[0].set_index(0);

  m_initializer->init(builder.compile());
  auto ws = m_initializer->get_weight_params();
  auto bs = m_initializer->get_bias_params();
  auto dws = m_initializer->get_delta_weight_params();
  auto dbs = m_initializer->get_delta_bias_params();

  for (size_t i = 1; i < builder.get_total(); i++) {
    auto new_layer = Layer(m_layers[i]);
    auto layer_weight = ws[i - 1];
    auto layer_bias = bs[i - 1];
    auto layer_delta_weight = dws[i - 1];
    auto layer_delta_bias = dbs[i - 1];

    new_layer.set_params(layer_weight, layer_bias);
    new_layer.set_delta_params(layer_delta_weight, layer_delta_bias);
    new_layer.set_index(i);

    if (!new_layer.regularization.are_created) {
      new_layer.regularization.m_l2Matrix =
          Eigen::MatrixXd::Zero(layer_weight.rows(), layer_weight.cols())
              .array() +
          new_layer.l2;
      new_layer.regularization.m_l2Vector =
          Eigen::VectorXd::Zero(layer_weight.rows()).array() + new_layer.l2;
      new_layer.regularization.are_created = true;
    }

    new_layer.set_previous(&m_layers[i - 1]);

    m_layers[i] = new_layer;
  }
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
    layer.fit(m_optimizer);
  }
}

Eigen::MatrixXd Network::evaluate(const DataSplit::DataPoint &point) {
  auto Xs = point.X;
  auto Ys = point.Y;

  for (auto &layer : m_layers) {
    Xs = layer.calculate(Xs);
  }

  back_propagate(Ys);

  return Xs;
}

void Network::back_propagate(const Eigen::VectorXd &real) {
  Layer *last_layer = &m_layers[m_layers.size() - 1];
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
    eval = layer.calculate(eval);
  }

  return eval;
}
