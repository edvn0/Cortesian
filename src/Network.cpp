//
// Created by Edwin Carlsson on 2021-05-31.
//

#include "../include/Network.h"

Network::Network()
    : m_loss(nullptr), m_eval({nullptr}), m_optimizer(nullptr),
      m_initializer(nullptr) {
  this->operator()("loss", "None");
  this->operator()("evaluation", "None");
  this->operator()("optimizer", "None");
  this->operator()("initializer", "None");
}

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

    m_layers[i] = new_layer;
  }

  this->operator()("num_layers", std::to_string(m_layers.size()));
  std::string layers = "[";
  std::string key_layers = "layers";
  for (auto l : m_layers) {
    layers.append("{");
    for (auto &[k, v] : l->key_value_pairs())
      layers.append(k).append(":").append(v).append(",");
    layers.append("},");
  }
  layers.append("]");
  this->operator()(key_layers, layers);

  std::string eval = "{";
  for (auto l : m_eval) {
    for (auto &[k, v] : l->key_value_pairs())
      eval.append(k).append(":").append(v).append(";");
  }
  eval.append("}");
  this->operator()("evaluation", eval);

  std::string optim = "{";
  for (auto &[k, v] : m_optimizer->key_value_pairs())
    optim.append(k).append(":").append(v).append(";");
  optim.append("}");
  this->operator()("optimizer", optim);

  std::string loss = "{";
  for (auto &[k, v] : m_loss->key_value_pairs())
    loss.append(k).append(":").append(v).append(";");
  loss.append("}");
  this->operator()("loss", loss);

  std::string initializer = "{";
  for (auto &[k, v] : m_initializer->key_value_pairs())
    initializer.append(k).append(":").append(v).append(";");
  initializer.append("}");
  this->operator()("initializer", initializer);

  std::string clipping = "{";
  for (auto &[k, v] : m_clipping.key_value_pairs())
    clipping.append(k).append(":").append(v).append(";");
  clipping.append("}");
  this->operator()("clipping", clipping);
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
  std::string json = Network::to_json(network);
  return os << json;
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
  size_t size = ds.rows.size();
  for (int i = 0; i < size; i++) {
    evaluate_for_back_prop(ds.rows[i]);
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

  int layer = last_layer->get_layer_index();
  do {
    auto previous_layer = m_layers[layer - 1];
    auto dc_di = last_layer->error_derivative(z, gradient);

    auto previous_activation = previous_layer->get_activated().transpose();

    Eigen::MatrixXd d_w = dc_di * previous_activation;

    last_layer->add_deltas(d_w, dc_di);

    gradient = last_layer->get_weight().transpose() * dc_di;

    last_layer = previous_layer;
    z = last_layer->get_activated();
    layer--;
  } while (layer > 0);
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
    Random::shuffle(splits);
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

    print_epoch_information(loss, metrics);
    backPropStatistics.update(timeEpoch, loss, metrics);
  }
  backPropStatistics.finalize();
  return backPropStatistics;
}

void Network::print_epoch_information(double loss,
                                      const std::vector<double> &metrics) {
  std::string epoch_output;
  epoch_output.reserve(70);
  epoch_output.append("Loss: ");
  epoch_output.append(std::to_string(loss));
  epoch_output.append(", Evaluation Scores: [");
  size_t metrics_size = metrics.size();
  for (auto &val : metrics) {
    epoch_output.append(std::to_string(val));
    if (metrics_size-- > 1)
      epoch_output.append(", ");
  }
  epoch_output.append("]\n");
  std::cout << epoch_output;
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

void Network::save(const std::string &file_name) const {

  for (auto &layer : m_layers) {
    layer->serialize_parameters();
  }

  std::fstream file(file_name, std::fstream::out);
  std::string json = Network::to_json(*this);
  file.write(json.data(), (long)json.size());
  file.close();
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

std::string Network::to_json(const Network &network) {
  std::string json;
  json.reserve(500);
  json.append("{");
  json.append("\n\t\"Optimizer\": {");
  auto optim_data = network.m_optimizer->key_value_pairs();
  int max_data_size = optim_data.size() - 1;
  int j = 0;
  std::for_each(
      optim_data.begin(), optim_data.end(),
      [&json, &j, &max_data_size](std::tuple<std::string, std::string> &s) {
        json.append("\n\t\t")
            .append("\"")
            .append(std::get<0>(s))
            .append("\": \"")
            .append(std::get<1>(s))
            .append("\"");

        if (j != max_data_size) {
          json.append(",");
        }

        j++;
      });
  json.append("\n\t},");

  json.append("\n\t\"Loss\": {");
  auto loss_data = network.m_loss->key_value_pairs();
  max_data_size = loss_data.size() - 1;
  j = 0;
  std::for_each(
      loss_data.begin(), loss_data.end(),
      [&json, &j, &max_data_size](std::tuple<std::string, std::string> &s) {
        json.append("\n\t\t")
            .append("\"")
            .append(std::get<0>(s))
            .append("\": \"")
            .append(std::get<1>(s))
            .append("\"");

        if (j != max_data_size) {
          json.append(",");
        }
        j++;
      });
  json.append("\n\t},");

  json.append("\n\t\"Evaluators\": [");
  int i = 0;
  int eval_size = network.m_eval.size() - 1;
  for (auto eval : network.m_eval) {
    json.append("\n\t\t{");
    auto eval_data = eval->key_value_pairs();
    max_data_size = eval_data.size() - 1;
    j = 0;
    std::for_each(
        eval_data.begin(), eval_data.end(),
        [&json, &j, &max_data_size](std::tuple<std::string, std::string> &s) {
          json.append("\n\t\t\t")
              .append("\"")
              .append(std::get<0>(s))
              .append("\": \"")
              .append(std::get<1>(s))
              .append("\"");

          if (j != max_data_size) {
            json.append(",");
          }
          j++;
        });

    if (i != eval_size) {
      json.append("\n\t\t},");
    } else {
      json.append("\n\t\t}");
    }
    i++;
  }
  json.append("\n\t],");

  json.append("\n\t\"Initializer\": {");
  auto initializer_data = network.m_initializer->key_value_pairs();
  max_data_size = initializer_data.size() - 1;
  j = 0;
  std::for_each(
      initializer_data.begin(), initializer_data.end(),
      [&json, &j, &max_data_size](std::tuple<std::string, std::string> &s) {
        json.append("\n\t\t")
            .append("\"")
            .append(std::get<0>(s))
            .append("\": \"")
            .append(std::get<1>(s))
            .append("\"");

        if (j != max_data_size) {
          json.append(",");
        }

        j++;
      });
  json.append("\n\t},");

  json.append("\n\t\"Layers\": [");
  int layer_size = network.m_layers.size() - 1;
  i = 0;
  for (auto layer : network.m_layers) {
    json.append("\n\t\t{");
    auto layer_data = layer->key_value_pairs();
    max_data_size = layer_data.size() - 1;
    j = 0;
    std::for_each(
        layer_data.begin(), layer_data.end(),
        [&json, &j, &max_data_size](std::tuple<std::string, std::string> &s) {
          json.append("\n\t\t\t")
              .append("\"")
              .append(std::get<0>(s))
              .append("\": \"")
              .append(std::get<1>(s))
              .append("\"");

          if (j != max_data_size) {
            json.append(",");
          }

          j++;
        });

    if (i != layer_size) {
      json.append("\n\t\t},");
    } else {
      json.append("\n\t\t}");
    }
    i++;
  }
  json.append("\n\t]");

  json.append("\n}\n");
  return json;
}

Network Network::from_json(const std::string &json) {

  std::cout << json;

  NetworkBuilder builder;
  // TODO: needs to be implemented.
  // FIXME: needs to be implemented.
  return Network(builder);
}

Network Network::from_json_file(const std::string &file_path) {
  std::string json;
  std::fstream json_file(file_path, std::fstream::in);
  std::stringstream buffer;
  buffer << json_file.rdbuf();

  json = buffer.str();
  return from_json(json);
}
