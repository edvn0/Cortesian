//
// Created by Edwin Carlsson on 2021-06-01.
//

#ifndef CORTESIAN_EIGENINITIALIZER_H
#define CORTESIAN_EIGENINITIALIZER_H

#include <cmath>
#include <vector>

#include "ParameterInitializer.h"

class EigenInitializer : public ParameterInitializer {
private:
  static double glorot_limit(size_t neurons_in, size_t neurons_out) {
    return sqrt(6.0 / ((double)neurons_in + (double)neurons_out));
  }

public:
  EigenInitializer();

  std::vector<Eigen::MatrixXd> get_weight_params() override;
  std::vector<Eigen::VectorXd> get_bias_params() override;
  std::vector<Eigen::MatrixXd> get_delta_weight_params() override;
  std::vector<Eigen::VectorXd> get_delta_bias_params() override;
  void init(std::vector<int> structure) override;
};

#endif // CORTESIAN_EIGENINITIALIZER_H
