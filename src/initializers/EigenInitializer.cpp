//
// Created by Edwin Carlsson on 2021-06-01.
//

#include "../../include/initializers/EigenInitializer.h"

std::vector<Eigen::MatrixXd> EigenInitializer::get_weight_params() {
  std::vector<Eigen::MatrixXd> weights;
  for (size_t i = 0; i < m_offset_length; i++) {
    int current = m_structure[i + 1];
    int next = m_structure[i];

    Eigen::MatrixXd glorot = (Eigen::MatrixXd::Random(current, next).array()) *
                             glorot_limit(current, next);
    weights.emplace_back(glorot);
  }
  return weights;
}

std::vector<Eigen::VectorXd> EigenInitializer::get_bias_params() {
  std::vector<Eigen::VectorXd> bias;
  for (size_t i = 0; i < m_offset_length; i++) {
    int current = m_structure[i + 1];
    int next = m_structure[i];

    Eigen::VectorXd glorot = (Eigen::VectorXd::Random(current, 1).array()) *
                             glorot_limit(current, next);
    bias.emplace_back(glorot);
  }
  return bias;
}

std::vector<Eigen::MatrixXd> EigenInitializer::get_delta_weight_params() {
  std::vector<Eigen::MatrixXd> weights;
  for (size_t i = 0; i < m_offset_length; i++) {
    int current = m_structure[i + 1];
    int next = m_structure[i];

    Eigen::MatrixXd glorot = Eigen::MatrixXd::Zero(current, next);
    weights.emplace_back(glorot);
  }
  return weights;
}

std::vector<Eigen::VectorXd> EigenInitializer::get_delta_bias_params() {
  std::vector<Eigen::VectorXd> weights;
  for (size_t i = 0; i < m_offset_length; i++) {
    int current = m_structure[i + 1];
    int next = 1;

    Eigen::VectorXd glorot = Eigen::VectorXd::Zero(current, next);
    weights.emplace_back(glorot);
  }
  return weights;
}

void EigenInitializer::init(std::vector<int> structure) {
  m_is_initialized = true;
  m_structure = structure;
  m_offset_length = m_structure.size() - 1;
}
EigenInitializer::EigenInitializer() {
  this->operator()("initializer", "EigenInitializer");
}
